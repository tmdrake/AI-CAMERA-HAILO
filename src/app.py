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
    
    while running:
        try:
            if camera is None:
                time.sleep(0.1)
                continue
            
            frame = camera.capture_array()
            
            with latest_frame_lock:
                latest_frame = frame.copy()
            
            if detector:
                threshold = config.get('detection.confidence_threshold', 0.5)
                detections = detector.detect(frame, threshold)
                
                if detections:
                    event_handler.handle_detection(detections, frame, datetime.now())
            
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
            elif key == 'alerts.enabled':
                config.set('alerts.enabled', request.form.get('alerts.enabled') == 'on')
            elif key == 'recording.enabled':
                config.set('recording.enabled', request.form.get('recording.enabled') == 'on')
            elif key.startswith('alerts.email.'):
                config.set(key, value)
            elif key.startswith('web_server.'):
                config.set(key, value)
        
        config.save()
        return redirect(url_for('settings'))
    
    return render_template('settings.html', config=config.get_all())

@app.route('/events')
@login_required
def events():
    events = event_handler.get_recent_events(50)
    return render_template('events.html', events=events)

@app.route('/api/events')
@login_required
def api_events():
    events = event_handler.get_recent_events(50)
    return jsonify([{
        'timestamp': e.timestamp.isoformat(),
        'detections': [{'class_name': d.class_name, 'confidence': d.confidence} for d in e.detections],
        'image_path': e.image_path
    } for e in events])

@app.route('/api/status')
@login_required
def api_status():
    return jsonify({
        'running': running,
        'detections': len(event_handler.events) if event_handler else 0,
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
