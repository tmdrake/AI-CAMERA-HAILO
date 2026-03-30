#!/usr/bin/env python3

import os
import sys
import threading
import time
import logging
import io
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Add system packages for hailo_platform
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import numpy as np
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from src.config import config
from src.detector import create_detector, Detection
from src.events import EventHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'web', 'templates'))
app.secret_key = 'ai-camera-secret-key-change-in-production'

@app.context_processor
def inject_config():
    return dict(config=config.get_all())

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

detector = None
event_handler = None
camera = None
running = False
detection_thread = None
latest_frame = None
latest_frame_lock = threading.Lock()

def init_camera():
    global camera
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        
        width = config.get('camera.width', 1920)
        height = config.get('camera.height', 1080)
        framerate = config.get('camera.framerate', 30)
        
        camera_config = camera.create_video_configuration(
            main={"size": (width, height)},
            controls={"FrameRate": framerate}
        )
        camera.configure(camera_config)
        camera.start()
        
        logger.info("Camera initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return False

def detection_loop():
    global latest_frame, running
    
    frame_count = 0
    detection_count = 0
    
    while running:
        try:
            if camera is None:
                time.sleep(0.1)
                continue
            
            frame = camera.capture_array()
            
            with latest_frame_lock:
                latest_frame = frame.copy()
            
            frame_count += 1
            
            if detector:
                threshold = config.get('detection.confidence_threshold', 50) / 100.0
                detections = detector.detect(frame, threshold)
                
                if detections:
                    logger.info(f"Threshold: {threshold}, Detections: {len(detections)}")
                
                if detections and event_handler:
                    logger.info(f"EVENT TRIGGERED: {len(detections)} detections!")
                    for d in detections:
                        logger.info(f"  {d.class_name}: conf={d.confidence:.2f} at {d.bbox}")
                    event_handler.handle_detection(detections, frame, datetime.now())
                    detection_count += 1
            
            # Log every 300 frames (~10 seconds)
            if frame_count % 300 == 0:
                logger.info(f"Processed {frame_count} frames, {detection_count} detections")
            
            time.sleep(0.03)
            
        except Exception as e:
            logger.error(f"Detection loop error: {e}")
            time.sleep(1)

def generate_frames():
    global latest_frame
    
    while running:
        try:
            with latest_frame_lock:
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                frame = latest_frame.copy()
            
            import cv2
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame_rgb, encode_param)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            time.sleep(0.1)

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/live')
@login_required
def live():
    return render_template('live.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        for key in request.form.keys():
            value = request.form.get(key)
            
            if key == 'detection.confidence_threshold':
                config.set('detection.confidence_threshold', float(value))
            elif key == 'alerts.email.enabled':
                config.set('alerts.email.enabled', request.form.get('alerts.email.enabled') == 'on')
            elif key == 'alerts.email.use_tls':
                config.set('alerts.email.use_tls', request.form.get('alerts.email.use_tls') == 'on')
            elif key == 'alerts.enabled':
                config.set('alerts.enabled', request.form.get('alerts.enabled') == 'on')
            elif key == 'recording.enabled':
                config.set('recording.enabled', request.form.get('recording.enabled') == 'on')
            elif key == 'alerts.email.recipient_emails':
                emails = [e.strip() for e in value.split(',') if e.strip()]
                config.set('alerts.email.recipient_emails', emails)
            elif key == 'alerts.email.smtp_port':
                config.set('alerts.email.smtp_port', int(value))
            elif key == 'alerts.cooldown_seconds':
                config.set('alerts.cooldown_seconds', int(value))
            elif key.startswith('alerts.email.'):
                config.set(key, value)
            elif key.startswith('web_server.'):
                config.set(key, value)
        
        config.save()
        return redirect(url_for('settings'))
    
    return render_template('settings.html', config=config.get_all())

@app.route('/settings/reset')
@login_required
def factory_reset():
    global running
    
    logger.info("Factory reset requested")
    
    DEFAULT_CONFIG = {
        "device": {
            "name": "AI Camera"
        },
        "detection": {
            "confidence_threshold": 50,
            "person_class_only": True,
            "model_path": "/usr/share/hailo-models/yolov5s_personface_h8l.hef"
        },
        "recording": {
            "enabled": True,
            "storage_path": "recordings",
            "retention_days": 90,
            "record_duration_seconds": 30,
            "pre_buffer_seconds": 5
        },
        "alerts": {
            "enabled": True,
            "email": {
                "enabled": False,
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "sender_email": "",
                "sender_password": "",
                "recipient_emails": []
            },
            "cooldown_seconds": 60
        },
        "web_server": {
            "host": "0.0.0.0",
            "port": 5000,
            "username": "admin",
            "password": "admin123"
        },
        "camera": {
            "width": 640,
            "height": 480,
            "framerate": 30,
            "rotation": 0
        },
        "schedule": {
            "enabled": False,
            "active_hours": [
                {"start": "00:00", "end": "23:59"}
            ]
        }
    }
    
    for section, values in DEFAULT_CONFIG.items():
        for key, value in values.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    config.set(f"{section}.{key}.{subkey}", subvalue)
            else:
                config.set(f"{section}.{key}", value)
    
    config.save()
    
    logger.info("Factory reset complete - restarting")
    return redirect(url_for('settings'))

@app.route('/events')
@login_required
def events():
    events = event_handler.get_recent_events(50)
    return render_template('events.html', events=events)

@app.route('/api/events')
@login_required
def api_events():
    events = event_handler.get_recent_events(50)
    events = list(reversed(events))
    return jsonify([{
        'timestamp': e.timestamp.isoformat(),
        'detections': [{'class_name': d.class_name, 'confidence': d.confidence} for d in e.detections],
        'image_path': e.image_path,
        'image_filename': os.path.basename(e.image_path) if e.image_path else None
    } for e in events])

@app.route('/images/<filename>')
@login_required
def serve_image(filename):
    from flask import send_from_directory
    storage_path = config.get('recording.storage_path', 'recordings')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, storage_path)
    return send_from_directory(full_path, filename)

@app.route('/api/test-email', methods=['POST'])
@login_required
def test_email():
    if event_handler:
        result = event_handler.send_test_email()
        return jsonify(result)
    return jsonify({'success': False, 'error': 'Event handler not available'})

@app.route('/api/status')
@login_required
def api_status():
    if detector is None:
        detector_type = "None"
    elif hasattr(detector, '_initialize'):
        detector_type = "HailoDetector"
    else:
        detector_type = "MockDetector"
    
    last_event_time = None
    if event_handler and event_handler.events:
        last_event_time = event_handler.events[-1].timestamp.isoformat()
    
    return jsonify({
        'running': running,
        'detector_type': detector_type,
        'detections': len(event_handler.events) if event_handler else 0,
        'last_event_time': last_event_time,
        'cooldown': config.get('alerts.cooldown_seconds', 60),
        'threshold': config.get('detection.confidence_threshold', 50),
        'recording_enabled': config.get('recording.enabled', True),
        'camera_present': camera is not None,
        'config': config.get_all()
    })

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        expected_user = config.get('web_server.username', 'admin')
        expected_pass = config.get('web_server.password', 'admin123')
        
        if username == expected_user and password == expected_pass:
            login_user(User(username))
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def start():
    global detector, event_handler, running, detection_thread
    
    logger.info("Starting AI Camera...")
    
    model_path = config.get('detection.model_path')
    detector = create_detector(model_path)
    
    if detector and hasattr(detector, '_initialize'):
        logger.info("Using HailoDetector")
    else:
        logger.warning("Using MockDetector - no AI detection will occur")
    
    event_handler = EventHandler()
    
    if not init_camera():
        logger.warning("Camera initialization failed, running in preview mode")
    
    running = True
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    host = config.get('web_server.host', '0.0.0.0')
    port = config.get('web_server.port', 5000)
    
    logger.info(f"Starting web server on {host}:{port}")
    app.run(host=host, port=port, threaded=True)

def stop():
    global running, camera, detector
    
    logger.info("Stopping AI Camera...")
    running = False
    
    if camera:
        camera.close()
    
    if detector:
        detector.close()

if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt:
        stop()
