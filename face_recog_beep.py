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
import simpleaudio as sa
from urllib.parse import unquote

# EmailNotifier fallback
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
    def __init__(self, similarity_threshold=0.4, target_process_fps=10, beep_path="beep.wav"):
        self.similarity_threshold = similarity_threshold
        self.app = None
        self.embeddings_db = {}
        self.cap = None
        self.ffmpeg_process = None
        self.frame_queue = Queue(maxsize=30)
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

        # preload beep sound
        self.beep_wave = None
        try:
            if os.path.exists(beep_path):
                self.beep_wave = sa.WaveObject.from_wave_file(beep_path)
                print(f"Loaded beep sound: {beep_path}")
            else:
                print(f"Beep file not found: {beep_path}")
        except Exception as e:
            print("Failed to load beep:", e)
            self.beep_wave = None

        self._init_model()
        self._load_embeddings()

    def play_beep(self):
        if self.beep_wave is None:
            return
        try:
            def _play():
                try:
                    self.beep_wave.play()
                except Exception as e:
                    print("Beep playback failed:", e)
            threading.Thread(target=_play, daemon=True).start()
        except Exception as e:
            print("Beep thread error:", e)

    def _init_model(self):
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
        # decode percent-encoded characters
        rtsp_url = unquote(rtsp_url)
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-f", "image2pipe",
            "-s", f"{width}x{height}",
            "-r", str(out_fps),
            "-"
        ]
        print("FFmpeg cmd:", " ".join(cmd))
        return cmd

    def _ffmpeg_reader(self, width=640, height=480):
        frame_size = width * height * 3
        try:
            stdout = self.ffmpeg_process.stdout
            while (not self.reader_stop_event.is_set()) and (self.ffmpeg_process and self.ffmpeg_process.poll() is None):
                raw = stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    print("FFmpeg: incomplete frame or no data received. Triggering reconnect...")
                    break
                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                except Exception as e:
                    print("FFmpeg: frame reshape error:", e)
                    continue
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    try:
                        _ = self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except Empty:
                        pass
                self.last_frame_time = time.time()
        except Exception as e:
            print("FFmpeg reader thread exception:", e)
        finally:
            self.reader_stop_event.set()
            print("FFmpeg reader stopped.")

    def _start_ffmpeg(self, rtsp_url, width=640, height=480, out_fps=10):
        self.rtsp_url = rtsp_url
        cmd = self._build_ffmpeg_cmd(rtsp_url, width, height, out_fps)
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # capture errors for debug
                bufsize=10**7
            )
        except Exception as e:
            print("Failed to start ffmpeg:", e)
            self.ffmpeg_process = None
            return False

        self.reader_stop_event.clear()
        self.reader_thread = threading.Thread(target=self._ffmpeg_reader, args=(width, height), daemon=True)
        self.reader_thread.start()

        start = time.time()
        timeout = 8
        while time.time() - start < timeout:
            if not self.frame_queue.empty():
                print("FFmpeg feed started successfully.")
                self.last_frame_time = time.time()
                return True
            if self.ffmpeg_process.poll() is not None:
                err = self.ffmpeg_process.stderr.read().decode()
                print("FFmpeg error:", err)
                break
            time.sleep(0.1)

        print("No frames arrived within timeout.")
        return False

    def cleanup_ffmpeg(self):
        try:
            self.reader_stop_event.set()
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=1.0)
        except:
            pass
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                try: self.ffmpeg_process.kill()
                except: pass
            finally:
                self.ffmpeg_process = None
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: break

    def connect_camera(self, camera_type="webcam", rtsp_url=None):
        self.camera_type = camera_type
        if camera_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            return self.cap.isOpened()
        elif camera_type == "rtsp":
            if rtsp_url is None:
                rtsp_url = input("Enter RTSP URL: ").strip()
            self.rtsp_url = rtsp_url
            out_fps = min(10, self.target_process_fps)
            ok = self._start_ffmpeg(rtsp_url, width=640, height=480, out_fps=out_fps)
            if not ok:
                print("FFmpeg failed to produce frames.")
            return ok
        return False

    def reconnect_rtsp(self):
        print("Reconnecting RTSP...")
        self.cleanup_ffmpeg()
        time.sleep(1.5)
        if self.rtsp_url:
            return self.connect_camera("rtsp", rtsp_url=self.rtsp_url)
        else:
            print("No RTSP URL stored. Reconnect failed.")
            return False

    def get_frame(self, timeout=1.0):
        if self.camera_type == "rtsp":
            try:
                return self.frame_queue.get(timeout=timeout)
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
            color = (0,255,0) if name != "Unknown" else (0,0,255)
            cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
            label = f"{name}: {confidence:.2f}"
            cv2.putText(image, label, (max(x1,5), max(y1-10,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

    def record_attendance(self, name):
        if name == "Unknown":
            return
        current_time = time.time()
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
            except: pass

        if not os.path.exists('attendance.csv'):
            pd.DataFrame(columns=['emp_id','name','email','checkin','checkout','status']).to_csv('attendance.csv', index=False)
        try:
            df = pd.read_csv('attendance.csv')
        except:
            df = pd.DataFrame(columns=['emp_id','name','email','checkin','checkout','status'])

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_hour = datetime.now().hour

        today_records = df[(df['name']==name) & (df['checkin'].astype(str).str.startswith(current_date))]

        if today_records.empty:
            new_record = {'emp_id':emp_id,'name':name,'email':email,'checkin':timestamp,'checkout':'','status':'IN'}
            df = pd.concat([df,pd.DataFrame([new_record])], ignore_index=True)
            print(f"Check-in: {name} ({emp_id}) at {timestamp}")
            self.play_beep()
            try:
                threading.Thread(target=self.email_notifier.send_attendance_email, args=(email,name,emp_id,'IN',timestamp), daemon=True).start()
            except: pass
        elif current_hour>=17:
            latest_idx = today_records.index[-1]
            if df.loc[latest_idx,'status']=='IN':
                df.loc[latest_idx,'checkout']=timestamp
                df.loc[latest_idx,'status']='OUT'
                print(f"Check-out: {name} ({emp_id}) at {timestamp}")
                self.play_beep()
                try:
                    threading.Thread(target=self.email_notifier.send_attendance_email, args=(email,name,emp_id,'OUT',timestamp), daemon=True).start()
                except: pass

        df.to_csv('attendance.csv', index=False)

    def start_recognition(self):
        print("🎯 Face Recognition Attendance System (stable mode)")
        print("Select camera: 1=Webcam, 2=RTSP")
        choice = input("Choice (1-2): ").strip()
        camera_map = {"1":"webcam","2":"rtsp"}
        camera_type = camera_map.get(choice,"webcam")

        if camera_type=="rtsp":
            rtsp_url = input("Enter RTSP URL: ").strip()
            ok = self.connect_camera("rtsp", rtsp_url=rtsp_url)
        else:
            ok = self.connect_camera("webcam")

        if not ok:
            print("Camera connection failed")
            return

        print("Camera connected. Press 'q' to quit")

        try:
            while True:
                if self.camera_type=="rtsp" and self.reader_stop_event.is_set():
                    print("Reader stopped -> attempting reconnect...")
                    self.reader_stop_event.clear()
                    success = self.reconnect_rtsp()
                    if not success:
                        print("Reconnect failed. Waiting 5s...")
                        time.sleep(5)
                        continue
                    else:
                        print("Reconnected successfully.")
                        continue

                if self.camera_type=="rtsp" and time.time()-self.last_frame_time>12:
                    print("No frames for 12s, reconnecting...")
                    self.reconnect_rtsp()
                    continue

                frame = self.get_frame(timeout=1.0)
                if frame is None:
                    continue

                now = time.time()
                if now-self.last_process_time<self.process_interval:
                    display_frame = cv2.resize(frame,(640,480))
                    cv2.putText(display_frame,"Processing throttled",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                    cv2.imshow('Face Recognition', display_frame)
                    if cv2.waitKey(1)&0xFF==ord('q'): break
                    continue

                self.last_process_time=now

                small_frame = cv2.resize(frame,(320,240))
                faces = self.detect_faces(small_frame)
                face_results=[]

                for embedding, bbox in faces:
                    x1,y1,x2,y2=bbox
                    x1,y1,x2,y2=int(x1*2),int(y1*2),int(x2*2),int(y2*2)
                    name, confidence=self.recognize_face(embedding)
                    face_results.append((embedding,(x1,y1,x2,y2),name,confidence))
                    self.record_attendance(name)

                if not faces:
                    frame_to_show = frame.copy()
                    cv2.putText(frame_to_show,"No faces detected",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),2)
                else:
                    frame_to_show = self.draw_results(frame,face_results)

                cv2.putText(frame_to_show,"Press 'q' to quit",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.imshow('Face Recognition', frame_to_show)

                if cv2.waitKey(1)&0xFF==ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        if self.cap:
            try: self.cap.release()
            except: pass
        self.cleanup_ffmpeg()
        cv2.destroyAllWindows()
        print("Exited cleanly.")

if __name__=="__main__":
    system = FaceRecognitionSystem(similarity_threshold=0.4, target_process_fps=10)
    system.start_recognition()
