# AI Camera Project Plan

## Goal

Build a lightweight AI-powered security camera using:

-   Raspberry Pi + CSI Camera (RPI Camera)
-   Hailo AI accelerator (Hailo-8 or Hailo-8L), Drivers need to be installed and tested.
-   Person/Human detection only
-   Image recording on detection (Store locally like 90 days)
-   Email alerts (send Image)
-   Web UI with live stream and configurable settings (persisted)

**Constraints**: No desktop environment. All processing on-device.

## High-Level Architecture

```
RPi Camera → Hailo Inference → Detection Logic → Actions
                                     ↓
                               Web Server (Flask/FastAPI)
                                     ↓
                             Live Stream + Settings UI
```

Components:

-   **Core**: Python + Hailo SDK + rpi-Camera
-   **Detection**: YOLO or similar model optimized for Hailo (person class only)
-   **Alerts**: SMTP email with snapshot
-   **Frontend**: Simple web interface (HTML/JS + Bootstrap or minimal framework)
-   **Config**: JSON or YAML for settings (threshold, email, schedule, etc.)

## Implementation Phases

### Phase 1: Project Setup & Hardware Testing (Remote SSH)

**Note**: Developing code on Windows host in `AI-CAMERA/` then deploy to `~/AI-Camera` on ai-rpi via SSH/scp.

-   [x] Create directory structure
-   [ ] SSH to ai-rpi (dennisl), work in \~/AI-Camera
-   [ ] Test CSI camera module (libcamera-hello + capture test)
-   [ ] Verify Hailo SDK/drivers per RPi AI docs: confirm `dtparam=pciex1_gen=3`, install `hailo-all`
-   [ ] Setup virtual environment + requirements.txt 
-   [ ] Add configuration management (JSON for threshold/email/schedule)
-   [ ] Basic camera test script

### Phase 2: AI Detection Pipeline

-   [ ] Integrate Hailo SDK
-   [ ] Load optimized person detection model
-   [ ] Real-time inference loop with picamera2
-   [ ] Filter detections to humans only

### Phase 3: Event Handling

-   [ ] Trigger recording on detection
-   [ ] Implement email alerts with attachments
-   [ ] Add cooldown/debounce logic

### Phase 4: Web Interface

-   [ ] Live video streaming (MJPEG or WebRTC)
-   [ ] Settings page (sensitivity, email config, enable/disable)
-   [ ] Dashboard showing recent events
-   [ ] Settings persistence

### Phase 5: Polish & Deployment

-   [ ] System service (systemd)
-   [ ] Logging
-   [ ] Performance optimization

## Success Criteria

-   Reliable person detection with low false positives
-   Automatic recording + email on detection
-   Accessible web UI from local network
-   Configurable via web without code changes
-   Runs efficiently on Raspberry Pi + Hailo



\*\*  
These are needed for HAILO drivers:

sudo apt install dkms  
sudo apt install hailo-all

More information: Hailo Drivers Installation.txt
