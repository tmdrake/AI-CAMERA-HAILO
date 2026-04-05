#!/usr/bin/env python3

import os
import sys
import threading
import time
import logging
import io
from collections import deque
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
running = False
latest_frame = None
latest_frame_lock = threading.Lock()
latest_detections = []
latest_detections_lock = threading.Lock()
frame_buffer = deque(maxlen=15)
frame_buffer_lock = threading.Lock()

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
    capture_delay = config.get('detection.capture_delay_frames', 3)
    
    while running:
        try:
            if camera is None:
                time.sleep(0.1)
                continue
            
            frame = camera.capture_array()
            frame_time = datetime.now()
            
            with latest_frame_lock:
                latest_frame = frame.copy()
            
            with frame_buffer_lock:
                frame_buffer.append({'frame': frame.copy(), 'time': frame_time})
            
            frame_count += 1
            
            if detector:
                threshold = config.get('detection.confidence_threshold', 50) / 100.0
                detections = detector.detect(frame, threshold)
                
                with latest_detections_lock:
                    global latest_detections
                    latest_detections = detections
                
                if detections and event_handler:
                    delayed_frame = None
                    delayed_time = frame_time
                    
                    with frame_buffer_lock:
                        buffer_size = len(frame_buffer)
                        if buffer_size > capture_delay:
                            delayed_entry = frame_buffer[-(capture_delay + 1)]
                            delayed_frame = delayed_entry['frame']
                            delayed_time = delayed_entry['time']
                        elif buffer_size > 0:
                            delayed_entry = frame_buffer[-1]
                            delayed_frame = delayed_entry['frame']
                            delayed_time = delayed_entry['time']
                        else:
                            delayed_frame = frame
                    
                    if delayed_frame is not None:
                        delayed_detections = detector.detect(delayed_frame, threshold)
                        
                        if delayed_detections:
                            final_detections = delayed_detections
                            final_frame = delayed_frame
                        else:
                            final_detections = detections
                            final_frame = frame
                        
                        event_handler.handle_detection(final_detections, final_frame, delayed_time)
                    
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
            
            # Draw bounding boxes on frame
            with latest_detections_lock:
                detections = latest_detections
            
            # Draw ROI rectangle - visual indicator using current config
            height, width = frame.shape[:2]
            roi_percent = config.get('detection.roi_percent', 80.0) / 100.0
            roi_size = int(min(width, height) * roi_percent)
            x_start = (width - roi_size) // 2
            y_start = (height - roi_size) // 2
            # Draw thick cyan rectangle for ROI with dynamic label
            cv2.rectangle(frame, (x_start, y_start), (x_start + roi_size, y_start + roi_size), 
                         (0, 255, 255), 4)
            cv2.putText(frame, f"DETECTION AREA ({int(roi_percent*100)}%)", (x_start, y_start - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if detections:
                for det in detections:
                    x, y, w, h = det.bbox
                    conf_percent = int(det.confidence * 100)
                    label = f"{det.class_name.upper()} {conf_percent}%"
                    if det.class_name == 'person':
                        color = (0, 255, 0)  # Green for person
                        thickness = 4
                        font_scale = 0.9
                    else:
                        color = (255, 0, 255)  # Magenta for face
                        thickness = 5
                        font_scale = 1.1
                    # Draw with thicker lines and ensure box is visible
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                    cv2.putText(frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
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
            elif key == 'detection.roi_percent':
                config.set('detection.roi_percent', float(value))
            elif key == 'alerts.email.enabled':
                config.set('alerts.email.enabled', request.form.get('alerts.email.enabled') == 'on')
            elif key == 'alerts.email.use_tls':
                config.set('alerts.email.use_tls', request.form.get('alerts.email.use_tls') == 'on')
            elif key == 'alerts.enabled':
                config.set('alerts.enabled', request.form.get('alerts.enabled') == 'on')
            elif key == 'recording.enabled':
                config.set('recording.enabled', request.form.get('recording.enabled') == 'on')
            elif key == 'recording.retention_days':
                config.set('recording.retention_days', int(value))
            elif key == 'recording.record_duration_seconds':
                config.set('recording.record_duration_seconds', int(value))
            elif key.startswith('recording.'):
                config.set(key, value)
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
            "model_path": "/usr/share/hailo-models/yolov5s_personface_h8l.hef",
            "capture_delay_frames": 10
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
