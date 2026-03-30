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

## Hardware

This machine has a **Hailo-8 AI Accelerator** installed via PCIe.

## Implementation Phases

### Phase 1: Project Setup & Hardware Testing

-   [x] Create directory structure
-   [x] Test Hailo SDK installation - Working!
-   [x] Integrate Hailo Python API for inference
-   [x] Person detection model: yolov5s_personface_h8l.hef (uses Hailo8L model on Hailo8)
-   [x] Test with camera input

### Phase 2: AI Detection Pipeline

-   [x] Integrate Hailo SDK (using Python API with VDevice)
-   [x] Load optimized person detection model
-   [x] Real-time inference loop working
-   [x] Filter detections to humans only

### Phase 3: Event Handling

-   [x] Trigger recording on detection
-   [x] Implement email alerts with attachments
-   [x] Add cooldown/debounce logic

### Phase 4: Web Interface

-   [x] Live video streaming (MJPEG)
-   [x] Settings page (sensitivity, email config, enable/disable)
-   [x] Dashboard showing recent events
-   [x] Settings persistence

### Phase 5: Polish & Deployment

-   [x] System service (systemd)
-   [x] Logging

## Success Criteria

-   Reliable person detection with low false positives
-   Automatic recording + email on detection
-   Accessible web UI from local network
-   Configurable via web without code changes
-   Runs efficiently on Raspberry Pi + Hailo

## Hailo Python API Usage

```python
from hailo_platform import HEF, VDevice
import numpy as np

# Load model
hef = HEF('/usr/share/hailo-models/yolov5s_personface_h8l.hef')
with VDevice() as target:
    infer_model = target.create_infer_model(hef_path, 'yolov5s_personface')
    with infer_model.configure() as configured_model:
        configured_model.activate()
        
        input_name = infer_model.input_names[0]
        output_name = infer_model.output_names[0]
        
        input_buffer = np.zeros((640, 640, 3), dtype=np.uint8)
        output_buffer = np.zeros((802,), dtype=np.float32)
        
        bindings = configured_model.create_bindings()
        bindings.input(input_name).set_buffer(input_buffer)
        bindings.output(output_name).set_buffer(output_buffer)
        
        # Run inference
        configured_model.run([bindings], 1000)
        
        # Parse output: [num_dets, class_id, conf, x1, y1, x2, y2, ...]
        num_dets = int(output_buffer[0])
```
