# AI Camera - Lightweight AI-Powered Security Camera

A lightweight AI-powered security camera using Raspberry Pi + Hailo AI accelerator with person detection, recording, email alerts, and web UI.

## Project Structure

```
AI-Camera/
├── config/
│   └── settings.json          # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── detector.py            # Hailo AI detector
│   ├── events.py              # Event handling (recording, email)
│   ├── app.py                 # Main Flask application
│   └── camera_test.py         # Camera test script
├── web/
│   └── templates/              # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── live.html
│       ├── events.html
│       ├── settings.html
│       └── login.html
├── recordings/                 # Storage for recorded images
├── logs/                       # Log files
├── requirements.txt           # Python dependencies
├── ai-camera.service          # Systemd service file
└── plan.md                   # Project plan

```

## Installation on Raspberry Pi

1. **Install Hailo drivers:**
   ```bash
   sudo apt update
   sudo apt install dkms
   sudo apt install hailo-all
   ```

2. **Verify Hailo installation:**
   ```bash
   hailofw scan
   ```

3. **Test camera:**
   ```bash
   libcamera-hello
   ```

4. **Setup virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Configure settings:**
   Edit `config/settings.json` with your preferences (especially email settings).

6. **Install systemd service:**
   ```bash
   sudo cp ai-camera.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable ai-camera
   sudo systemctl start ai-camera
   ```

## Access

- Web UI: `http://<raspberry-pi-ip>:5000`
- Default login: `admin` / `admin123` (change in settings)

## Features

- Person detection using Hailo AI accelerator
- Image recording on detection (90-day retention)
- Email alerts with snapshot
- Live video streaming
- Configurable settings via web UI
