#!/usr/bin/env python3
"""
Basic camera test script for AI Camera project
Tests CSI camera functionality using picamera2
"""

import time
from picamera2 import Picamera2

def test_camera():
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Configure for preview
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    print("Starting camera...")
    picam2.start()
    
    # Let it run for 5 seconds
    print("Camera running for 5 seconds...")
    time.sleep(5)
    
    print("Stopping camera...")
    picam2.stop()
    picam2.close()
    
    print("Camera test completed successfully!")

if __name__ == "__main__":
    test_camera()