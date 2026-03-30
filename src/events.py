import os
import time
import smtplib
import threading
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import List, Optional
from pathlib import Path

from src.config import config
from src.detector import Detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Event:
    def __init__(self, timestamp: datetime, detections: List[Detection], image_path: str = None):
        self.timestamp = timestamp
        self.detections = detections
        self.image_path = image_path

class EventHandler:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = config.get('alerts.cooldown_seconds', 60)
        self.lock = threading.Lock()
        self.events: List[Event] = []
    
    def handle_detection(self, detections: List[Detection], frame, frame_time: datetime) -> bool:
        if not detections:
            return False
        
        cooldown = config.get('alerts.cooldown_seconds', 60)
        
        with self.lock:
            current_time = time.time()
            
            if current_time - self.last_alert_time < cooldown:
                return False
            
            event = Event(
                timestamp=frame_time,
                detections=detections,
                image_path=None
            )
            
            if config.get('recording.enabled', True):
                image_path = self._save_image(frame, frame_time)
                if image_path:
                    event.image_path = image_path
            
            if config.get('alerts.email.enabled', False):
                self._send_email_alert(detections, frame)
            
            self.last_alert_time = current_time
            self.events.append(event)
            
            logger.info(f"Event triggered: {len(detections)} person(s) detected")
            
            return True
    
    def _save_image(self, frame, timestamp: datetime) -> Optional[str]:
        try:
            from PIL import Image
            import numpy as np
            
            storage_path = config.get('recording.storage_path', 'recordings')
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, storage_path)
            
            os.makedirs(full_path, exist_ok=True)
            
            filename = timestamp.strftime("%Y%m%d_%H%M%S") + ".jpg"
            filepath = os.path.join(full_path, filename)
            
            if isinstance(frame, np.ndarray):
                if frame.shape[2] == 3:
                    frame = frame[:, :, ::-1]
                image = Image.fromarray(frame)
                image.save(filepath, quality=85)
            
            logger.info(f"Image saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None
    
    def _send_email_alert(self, detections: List[Detection], frame, image_path: str = None):
        email_config = config.get('alerts.email', {})
        
        if not email_config.get('enabled', False):
            return
        
        sender = email_config.get('sender_email', '')
        password = email_config.get('sender_password', '')
        recipients = email_config.get('recipient_emails', [])
        
        if not sender or not password or not recipients:
            logger.warning("Email not configured")
            return
        
        def send_async():
            try:
                device_name = config.get('device.name', 'AI Camera')
                msg = MIMEMultipart()
                msg['From'] = sender
                msg['To'] = ', '.join(recipients)
                msg['Subject'] = f"{device_name} Alert - {len(detections)} Person(s) Detected"
                
                body = f"Motion detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                body += f"Detected {len(detections)} person(s)\n\n"
                body += "Confidence: " + ", ".join([f"{d.confidence:.2f}" for d in detections])
                
                msg.attach(MIMEText(body, 'plain'))
                
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                        msg.attach(img)
                else:
                    import numpy as np
                    from PIL import Image
                    if isinstance(frame, np.ndarray):
                        if frame.shape[2] == 3:
                            frame_rgb = frame[:, :, ::-1]
                        else:
                            frame_rgb = frame
                        img = Image.fromarray(frame_rgb)
                        import io
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='JPEG')
                        img_bytes = img_bytes.getvalue()
                        img_attachment = MIMEImage(img_bytes)
                        img_attachment.add_header('Content-Disposition', 'attachment', filename='alert.jpg')
                        msg.attach(img_attachment)
                
                smtp_host = email_config.get('smtp_host', 'smtp.gmail.com')
                smtp_port = email_config.get('smtp_port', 587)
                use_tls = email_config.get('use_tls', True)
                
                server = smtplib.SMTP(smtp_host, smtp_port)
                if use_tls:
                    server.starttls()
                if password:
                    server.login(sender, password)
                server.send_message(msg)
                server.quit()
                
                logger.info("Email alert sent successfully")
                
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
        
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()
    
    def send_test_email(self) -> dict:
        email_config = config.get('alerts.email', {})
        
        sender = email_config.get('sender_email', '')
        password = email_config.get('sender_password', '')
        recipients = email_config.get('recipient_emails', [])
        
        if not sender:
            return {'success': False, 'error': 'Sender email not configured'}
        
        if not recipients:
            return {'success': False, 'error': 'No recipient emails configured'}
        
        def send_async():
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                smtp_host = email_config.get('smtp_host', 'smtp.gmail.com')
                smtp_port = email_config.get('smtp_port', 587)
                use_tls = email_config.get('use_tls', True)
                
                msg = MIMEMultipart()
                msg['From'] = sender
                msg['To'] = ', '.join(recipients)
                device_name = config.get('device.name', 'AI Camera')
                msg['Subject'] = f'{device_name} - Test Email'
                msg.attach(MIMEText('This is a test email from your AI Camera system.', 'plain'))
                
                server = smtplib.SMTP(smtp_host, smtp_port)
                if use_tls:
                    server.starttls()
                if password:
                    server.login(sender, password)
                server.send_message(msg)
                server.quit()
                
                logger.info("Test email sent successfully")
                
            except Exception as e:
                logger.error(f"Failed to send test email: {e}")
                return {'success': False, 'error': str(e)}
        
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()
        
        return {'success': True, 'message': 'Test email sent'}
    
    def get_recent_events(self, limit: int = 50) -> List[Event]:
        with self.lock:
            return self.events[-limit:]
    
    def cleanup_old_recordings(self):
        try:
            retention_days = config.get('recording.retention_days', 90)
            storage_path = config.get('recording.storage_path', 'recordings')
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, storage_path)
            
            if not os.path.exists(full_path):
                return
            
            cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
            
            for filename in os.listdir(full_path):
                filepath = os.path.join(full_path, filename)
                if os.path.isfile(filepath):
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Removed old recording: {filename}")
                        
        except Exception as e:
            logger.error(f"Failed to cleanup recordings: {e}")
