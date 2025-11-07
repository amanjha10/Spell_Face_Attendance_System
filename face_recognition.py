import cv2
import numpy as np
import time
import subprocess
import threading
from queue import Queue, Empty
import pickle
import pandas as pd
from datetime import datetime
import os
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import pillow_heif
import signal

# EmailNotifier fallback if not available
try:
    from email_notifier import EmailNotifier
except ImportError:
    class EmailNotifier:
        def send_attendance_email(self, *args, **kwargs):
            print("Email notification disabled")
        def configure_smtp(self, *args, **kwargs):
            pass

pillow_heif.register_heif_opener()

class FaceRecognitionSystem:
    def __init__(self, similarity_threshold=0.4, target_process_fps=10):
        self.similarity_threshold = similarity_threshold
        self.app = None
        self.embeddings_db = {}
        self.cap = None
        self.ffmpeg_process = None
        self.frame_queue = Queue(maxsize=30)  # larger queue to absorb bursts
        self.camera_type = None
        self.last_attendance_time = {}
        self.email_notifier = EmailNotifier()
        self.rtsp_url = None
        self.reader_thread = None
        self.reader_stop_event = threading.Event()
        self.last_frame_time = 0.0
        self.last_process_time = 0.0
        self.target_process_fps = target_process_fps
        self.process_interval = 1.0 / max(1, self.target_process_fps)
        self._init_model()
        self._load_embeddings()

    def _init_model(self):
        # prepare insightface
        print("Loading InsightFace model ...")
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Model loaded.")

    def _load_embeddings(self):
        if os.path.exists('embeddings/faces.pkl'):
            try:
                with open('embeddings/faces.pkl', 'rb') as f:
                    data = pickle.load(f)
                    for name, info in data.items():
                        self.embeddings_db[name] = info['embeddings']
                print(f"Loaded embeddings for {len(self.embeddings_db)} identities.")
            except Exception as e:
                print("Failed to load embeddings:", e)

    def _build_ffmpeg_cmd(self, rtsp_url, width=640, height=480, out_fps=10):
        # -stimeout expects microseconds (here 5s = 5_000_000)
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-f", "image2pipe",
            "-s", "640x480",
            "-r", "15",
            "-"

        ]
        return cmd

    def _ffmpeg_reader(self, width=640, height=480):
        """
        Reader thread: reads raw frames from ffmpeg stdout and pushes into frame_queue.
        If incomplete frame is detected or ffmpeg exits, attempt reconnect by cleaning up
        and signaling the main loop via reader_stop_event.
        """
        frame_size = width * height * 3
        try:
            stdout = self.ffmpeg_process.stdout
            # continuously read frames
            while (not self.reader_stop_event.is_set()) and (self.ffmpeg_process and self.ffmpeg_process.poll() is None):
                # read exactly frame_size bytes
                raw = stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    # incomplete / missing frame — likely ffmpeg timed out or camera hiccup
                    print("FFmpeg: incomplete frame or no data received. Triggering reconnect...")
                    break  # break to reconnect
                # convert to numpy image
                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                except Exception as e:
                    print("FFmpeg: frame reshape error:", e)
                    continue

                # put into queue (drop oldest if full)
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    try:
                        _ = self.frame_queue.get_nowait()  # drop oldest
                        self.frame_queue.put_nowait(frame)
                    except Empty:
                        pass

                # update last_frame_time for watchdog
                self.last_frame_time = time.time()

            # reader loop ending -> do cleanup and notify main thread to reconnect
        except Exception as e:
            print("FFmpeg reader thread exception:", e)
        finally:
            # ensure ffmpeg process is cleaned and event set so main loop can reconnect
            self.reader_stop_event.set()
            # do not call reconnect here directly to avoid nested prompts/loops
            # main loop will notice reader_stop_event and handle reconnection
            # print for debug:
            print("FFmpeg reader stopped.")

    def _start_ffmpeg(self, rtsp_url, width=640, height=480, out_fps=10):
        """Start ffmpeg subprocess and reader thread. Returns True if frames start flowing."""
        self.rtsp_url = rtsp_url
        cmd = self._build_ffmpeg_cmd(rtsp_url, width, height, out_fps)
        try:
            # Redirect stderr to DEVNULL to avoid blocking on stderr buffer
            self.ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)
        except Exception as e:
            print("Failed to start ffmpeg:", e)
            self.ffmpeg_process = None
            return False

        # clear stop event and start reader thread
        self.reader_stop_event.clear()
        self.reader_thread = threading.Thread(target=self._ffmpeg_reader, args=(width, height), daemon=True)
        self.reader_thread.start()

        # wait a short time to check if frames arrive
        start = time.time()
        timeout = 8
        while time.time() - start < timeout:
            if not self.frame_queue.empty():
                print("FFmpeg feed started successfully.")
                self.last_frame_time = time.time()
                return True
            # also check if ffmpeg died immediately
            if self.ffmpeg_process.poll() is not None:
                break
            time.sleep(0.1)

        print("No frames arrived within timeout.")
        return False

    def cleanup_ffmpeg(self):
        # stop reader
        try:
            self.reader_stop_event.set()
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=1.0)
        except Exception:
            pass

        if self.ffmpeg_process:
            try:
                # try to terminate gracefully
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except Exception:
                try:
                    self.ffmpeg_process.kill()
                except Exception:
                    pass
            finally:
                self.ffmpeg_process = None

        # clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break

    def connect_camera(self, camera_type="webcam", rtsp_url=None):
        self.camera_type = camera_type
        if camera_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            return self.cap.isOpened()
        elif camera_type == "droidcam":
            ip = input("Enter DroidCam IP: ").strip() or "192.168.1.100"
            url = f"http://{ip}:4747/video"
            self.cap = cv2.VideoCapture(url)
            return self.cap.isOpened()
        elif camera_type == "rtsp":
            if rtsp_url is None:
                rtsp_url = input("Enter RTSP URL: ").strip()
            self.rtsp_url = rtsp_url
            # desired outgoing fps from ffmpeg (and thus queue)
            out_fps = min(10, self.target_process_fps)
            ok = self._start_ffmpeg(rtsp_url, width=640, height=480, out_fps=out_fps)
            if not ok:
                print("FFmpeg failed to produce frames.")
            return ok
        return False

    def reconnect_rtsp(self):
        print("Reconnecting RTSP...")
        try:
            self.cleanup_ffmpeg()
            time.sleep(1.5)
            # try to restart using stored rtsp_url
            if self.rtsp_url:
                return self.connect_camera("rtsp", rtsp_url=self.rtsp_url)
            else:
                print("No RTSP URL stored. Reconnect failed.")
                return False
        except Exception as e:
            print("Reconnect failed:", e)
            return False

    def get_frame(self, timeout=1.0):
        """
        Returns a frame if available within timeout, else None.
        """
        if self.camera_type == "rtsp":
            try:
                frame = self.frame_queue.get(timeout=timeout)
                return frame
            except Empty:
                return None
        else:
            if not self.cap:
                return None
            ret, frame = self.cap.read()
            return frame if ret else None

    def detect_faces(self, image):
        if image is None:
            return []
        try:
            faces = self.app.get(image)
            # return list of (embedding, bbox)
            return [(face.normed_embedding, face.bbox.astype(int)) for face in faces]
        except Exception as e:
            print("Face detect error:", e)
            return []

    def recognize_face(self, embedding):
        best_match = "Unknown"
        best_score = 0.0
        for name, embeddings in self.embeddings_db.items():
            for stored in embeddings:
                score = float(np.dot(embedding, stored))
                if score > best_score:
                    best_score = score
                    if score >= self.similarity_threshold:
                        best_match = name
        return best_match, best_score

    def draw_results(self, image, faces_results):
        image = image.copy()
        for embedding, bbox, name, confidence in faces_results:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{name}: {confidence:.2f}"
            cv2.putText(image, label, (max(x1,5), max(y1-10,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            status = f"{name} Recognized!" if name != "Unknown" else "Unknown Person"
            cv2.putText(image, status, (max(x1,5), y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

    def record_attendance(self, name):
        if name == "Unknown":
            return
        current_time = time.time()
        # debounce for 60s per user
        if name in self.last_attendance_time and current_time - self.last_attendance_time[name] < 60:
            return
        self.last_attendance_time[name] = current_time

        emp_id = "EMP001"
        email = f"{name.lower()}@example.com"
        if os.path.exists('embeddings/faces.pkl'):
            try:
                with open('embeddings/faces.pkl', 'rb') as f:
                    data = pickle.load(f)
                    if name in data:
                        emp_id = data[name].get('employee_id', emp_id)
                        email = data[name].get('email', email)
            except Exception:
                pass

        if not os.path.exists('attendance.csv'):
            pd.DataFrame(columns=['emp_id', 'name', 'email', 'checkin', 'checkout', 'status']).to_csv('attendance.csv', index=False)

        try:
            df = pd.read_csv('attendance.csv')
        except Exception:
            df = pd.DataFrame(columns=['emp_id', 'name', 'email', 'checkin', 'checkout', 'status'])

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_hour = datetime.now().hour

        today_records = df[(df['name'] == name) & (df['checkin'].astype(str).str.startswith(current_date))]

        if today_records.empty:
            new_record = {'emp_id': emp_id, 'name': name, 'email': email, 'checkin': timestamp, 'checkout': '', 'status': 'IN'}
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            print(f"Check-in: {name} ({emp_id}) at {timestamp}")
            threading.Thread(target=self.email_notifier.send_attendance_email, args=(email, name, emp_id, 'IN', timestamp), daemon=True).start()
        elif current_hour >= 17:
            latest_idx = today_records.index[-1]
            if df.loc[latest_idx, 'status'] == 'IN':
                df.loc[latest_idx, 'checkout'] = timestamp
                df.loc[latest_idx, 'status'] = 'OUT'
                print(f"🚪 Check-out: {name} ({emp_id}) at {timestamp}")
                threading.Thread(target=self.email_notifier.send_attendance_email, args=(email, name, emp_id, 'OUT', timestamp), daemon=True).start()

        df.to_csv('attendance.csv', index=False)

    def start_recognition(self):
        print("🎯 Face Recognition Attendance System (stable mode)")
        print("Select camera: 1=Webcam, 2=DroidCam, 3=RTSP")
        choice = input("Choice (1-3): ").strip()
        camera_map = {"1": "webcam", "2": "droidcam", "3": "rtsp"}
        camera_type = camera_map.get(choice, "webcam")

        if camera_type == "rtsp":
            rtsp_url = input("Enter RTSP URL: ").strip()
            ok = self.connect_camera("rtsp", rtsp_url=rtsp_url)
        else:
            ok = self.connect_camera(camera_type)

        if not ok:
            print("Camera connection failed")
            return

        print("Camera connected. Press 'q' to quit")

        try:
            while True:
                # If reader thread signaled a stop (ffmpeg stopped), attempt reconnect
                if self.camera_type == "rtsp" and self.reader_stop_event.is_set():
                    print("Reader stopped -> attempting reconnect...")
                    self.reader_stop_event.clear()
                    success = self.reconnect_rtsp()
                    if not success:
                        print("Reconnect failed. Waiting 5s before retry...")
                        time.sleep(5)
                        continue
                    else:
                        print("Reconnected successfully.")
                        continue

                # watchdog: if no frame received recently, reconnect
                if self.camera_type == "rtsp":
                    if time.time() - self.last_frame_time > 12:  # no frames for 12s
                        print("No frames for 12s, reconnecting...")
                        self.reconnect_rtsp()
                        continue

                frame = self.get_frame(timeout=1.0)
                if frame is None:
                    # no frame this cycle
                    continue

                now = time.time()
                # throttle processing so detector runs at ~target_process_fps
                if now - self.last_process_time < self.process_interval:
                    # we still want to show live feed though, so display
                    display_frame = cv2.resize(frame, (640, 480))
                    cv2.putText(display_frame, "Processing throttled (saving CPU)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Face Recognition', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                self.last_process_time = now

                # downscale for faster detection (model det_size was 640 so keep reasonable)
                small_frame = cv2.resize(frame, (320, 240))
                faces = self.detect_faces(small_frame)
                face_results = []

                for embedding, bbox in faces:
                    # bbox corresponds to small_frame scale; scale back to original
                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2 = int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)
                    name, confidence = self.recognize_face(embedding)
                    face_results.append((embedding, (x1, y1, x2, y2), name, confidence))
                    self.record_attendance(name)

                if not faces:
                    frame_to_show = frame.copy()
                    cv2.putText(frame_to_show, "No faces detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                else:
                    frame_to_show = self.draw_results(frame, face_results)

                cv2.putText(frame_to_show, "Live Face Recognition - Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Face Recognition', frame_to_show)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cleanup_ffmpeg()
        cv2.destroyAllWindows()
        print("Exited cleanly.")

if __name__ == "__main__":
    system = FaceRecognitionSystem(similarity_threshold=0.4, target_process_fps=10)
    system.start_recognition()
