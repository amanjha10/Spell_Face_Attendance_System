import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import pickle
import os


class EmailNotifier:
    def __init__(self):
        
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "amanjha2132@gmail.com"  
        self.sender_password = "lzmf wrtm aayt hplx"  
        self.company_name = "Spell Innovation"
    
    def configure_smtp(self, smtp_server=None, smtp_port=None, sender_email=None, sender_password=None):
        """Configure SMTP settings"""
        if smtp_server:
            self.smtp_server = smtp_server
        if smtp_port:
            self.smtp_port = smtp_port
        if sender_email:
            self.sender_email = sender_email
        if sender_password:
            self.sender_password = sender_password
    
    def send_attendance_email(self, to_email, person_name, emp_id, status, timestamp):
        """Send attendance notification email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = f"Attendance {status} - {person_name} ({emp_id})"
            
            
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            date_str = dt.strftime("%d %b %Y, %A")
            time_str = dt.strftime("%I:%M %p")
            
            
            if status == "IN":
                greeting = "Good Morning" if dt.hour < 12 else "Good Afternoon"
                body = f"""
Dear {person_name},

{greeting}!

Your attendance has been successfully marked as CHECK-IN for {date_str} at {time_str}.

Employee ID: {emp_id}
Date: {date_str}
Time: {time_str}
Status: Checked In ✅

Have a productive day ahead!

Best regards,
{self.company_name}
Dhani Ram Sapkota 
                """
            else:  # status == "OUT"
                body = f"""
Dear {person_name},

Your attendance has been successfully marked as CHECK-OUT for {date_str} at {time_str}.

Employee ID: {emp_id}
Date: {date_str}  
Time: {time_str}
Status: Checked Out 🚪

Thank you for your hard work today. Have a great evening!

Best regards,
{self.company_name}
Dhani Ram Sapkota
                """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"📧 Email sent to {person_name} ({to_email}) - {status}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email to {person_name}: {e}")
            return False
    
    def send_test_email(self, to_email, person_name="Test User"):
        """Send a test email to verify SMTP configuration"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = f"Test Email - Attendance System"
            
            body = f"""
Dear {person_name},

This is a test email from the Face Recognition Attendance System.

If you received this email, the email notification system is working correctly!

Best regards,
{self.company_name}
Attendance Management System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Test email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            print(f"❌ Test email failed: {e}")
            return False


def setup_email_config():
    """Interactive setup for email configuration"""
    print("📧 Email Notification Setup")
    print("=" * 40)
    
    notifier = EmailNotifier()
    
    print("Enter your SMTP configuration:")
    
    # Get SMTP server
    smtp_server = input(f"SMTP Server (default: {notifier.smtp_server}): ").strip()
    if smtp_server:
        notifier.smtp_server = smtp_server
    
    # Get SMTP port
    smtp_port = input(f"SMTP Port (default: {notifier.smtp_port}): ").strip()
    if smtp_port:
        notifier.smtp_port = int(smtp_port)
    
    # Get sender email
    sender_email = input("Your Email Address: ").strip()
    if sender_email:
        notifier.sender_email = sender_email
    
    # Get sender password
    print("\n📝 For Gmail, use App Password instead of regular password")
    print("   Go to: Google Account → Security → App passwords")
    sender_password = input("Email Password/App Password: ").strip()
    if sender_password:
        notifier.sender_password = sender_password
    
    # Test email
    test_email = input("\nTest email address (to verify setup): ").strip()
    if test_email:
        print("\n🧪 Sending test email...")
        if notifier.send_test_email(test_email):
            print("✅ Email configuration successful!")
        else:
            print("❌ Email configuration failed. Please check your settings.")
    
    return notifier


def get_person_email_from_embeddings(person_name):
    """Get person's email from embeddings file"""
    embeddings_file = 'embeddings/faces.pkl'
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                if person_name in data and 'email' in data[person_name]:
                    return data[person_name]['email']
        except:
            pass
    return None


if __name__ == "__main__":
    print("🎯 Email Notification System")
    print("=" * 40)
    print("1. 📧 Setup email configuration")
    print("2. 🧪 Send test email")
    print("3. 📊 View stored email addresses")
    print("4. 🚪 Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        setup_email_config()
    
    elif choice == '2':
        notifier = EmailNotifier()
        test_email = input("Enter test email address: ").strip()
        if test_email:
            notifier.send_test_email(test_email)
    
    elif choice == '3':
        embeddings_file = 'embeddings/faces.pkl'
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
            
            print("\n📧 Stored Email Addresses:")
            print("=" * 30)
            for name, info in data.items():
                email = info.get('email', 'Not set')
                emp_id = info.get('employee_id', 'Unknown')
                print(f"👤 {name} ({emp_id}): {email}")
        else:
            print("❌ No embeddings file found")
    
    elif choice == '4':
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")
