#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config

def test_camera():
    try:
        from picamera2 import Picamera2
        print("Testing camera...")
        
        picam2 = Picamera2()
        
        camera_config = picam2.create_still_configuration(
            main={"size": (config.get('camera.width', 1920), config.get('camera.height', 1080))}
        )
        picam2.configure(camera_config)
        
        picam2.start()
        
        print("Camera started successfully!")
        
        import time
        time.sleep(2)
        
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_capture.jpg")
        picam2.capture_file(output_path)
        
        print(f"Image captured to: {output_path}")
        
        picam2.close()
        
        return True
        
    except Exception as e:
        print(f"Camera test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
