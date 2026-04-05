import os
import sys
import numpy as np
from typing import List, Tuple, Optional
import logging
import cv2

# Import config for ROI settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Detection:
    def __init__(self, class_id: int, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox

class HailoDetector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "/usr/share/hailo-models/yolov5s_personface_h8l.hef"
        self.model_name = os.path.basename(model_path) if model_path else "unknown"
        self.vdevice = None
        self.infer_model = None
        self.configured_model = None
        self.bindings = None
        self.input_name = None
        self.output_name = None
        self._initialize()
    
    def _initialize(self):
        try:
            from hailo_platform import HEF, VDevice
            
            logger.info(f"Loading model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model not found: {self.model_path}")
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.vdevice = VDevice()
            self.infer_model = self.vdevice.create_infer_model(self.model_path)
            self.configured_model = self.infer_model.configure()
            
            self.configured_model.activate()
            
            self.input_name = self.infer_model.input_names[0]
            self.output_name = self.infer_model.output_names[0]
            
            self.input_shape = self.infer_model.inputs[0].shape
            self.output_shape = self.infer_model.outputs[0].shape
            
            self.input_buffer = np.zeros(self.input_shape, dtype=np.uint8)
            self.output_buffer = np.zeros(self.output_shape, dtype=np.float32)
            
            logger.info(f"Loaded model successfully")
            logger.info(f"Input: {self.input_name} shape={self.input_shape}, Output: {self.output_name} shape={self.output_shape}")
                    
        except ImportError as e:
            logger.error(f"Hailo platform not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        if self.configured_model is None:
            logger.error("Model not initialized")
            return []
        
        try:
            # Face-only mode with full frame for reliable detection
            orig_height, orig_width = frame.shape[:2]
            
            # Resize frame for model input
            resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
            np.copyto(self.input_buffer, resized)
            
            bindings = self.configured_model.create_bindings()
            bindings.input(self.input_name).set_buffer(self.input_buffer)
            bindings.output(self.output_name).set_buffer(self.output_buffer)
            
            self.configured_model.run([bindings], 1000)
            
            detections = self._parse_nms_output(self.output_buffer, confidence_threshold, (orig_height, orig_width), (self.input_shape[0], self.input_shape[1]))
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

    def _parse_nms_output(self, output: np.ndarray, confidence_threshold: float, orig_shape, model_shape, roi_size=None, x_offset=0, y_offset=0) -> List[Detection]:
        detections = []
        
        min_width = 25
        min_height = 25
        
        is_personface_model = 'personface' in self.model_name.lower()
        
        try:
            if output.size == 0:
                return detections
            
            num_dets = int(output[0])
            orig_height, orig_width = orig_shape[:2]
            
            # Use ROI parameters passed from detect() method
            if roi_size is None:
                roi_size = int(min(orig_width, orig_height) * 0.80)
                x_offset = (orig_width - roi_size) // 2
                y_offset = (orig_height - roi_size) // 2
            
            for i in range(min(num_dets, 50)):
                offset = 1 + i * 6
                if offset + 5 >= output.size:
                    break
                
                class_id = int(round(output[offset]))
                confidence = float(output[offset + 1])
                cx = float(output[offset + 2])
                cy = float(output[offset + 3])
                width = float(output[offset + 4])
                height = float(output[offset + 5])
                
                if width <= 0.01 or height <= 0.01:
                    continue
                
                if is_personface_model:
                    # Person only mode - only detect persons (class_id 1)
                    if class_id == 1 and confidence >= confidence_threshold:
                        class_name = 'person'
                    else:
                        continue
                else:
                    if class_id != 0 or confidence < confidence_threshold:
                        continue
                    class_name = 'person'
                
                cx = max(0.0, min(cx, 1.0))
                cy = max(0.0, min(cy, 1.0))
                width = max(0.01, min(width, 1.0))
                height = max(0.01, min(height, 1.0))
                
                # Convert normalized coordinates to pixel coordinates
                # Using center-width-height format from YOLO model
                x1 = int((cx - width/2) * roi_size + x_offset)
                y1 = int((cy - height/2) * roi_size + y_offset)
                w = int(height * roi_size)  # Note: width/height swapped to fix 90-degree rotation
                h = int(width * roi_size)
                
                x1 = max(0, min(x1, orig_width - 1))
                y1 = max(0, min(y1, orig_height - 1))
                w = min(w, orig_width - x1)
                h = min(h, orig_height - y1)
                
                if w < min_width or h < min_height:
                    continue
                
                bbox = (x1, y1, w, h)
                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox
                ))
            
        except Exception as e:
            logger.error(f"Failed to parse NMS output: {e}")
        
        return detections

    def close(self):
        try:
            if self.configured_model:
                self.configured_model.shutdown()
            if self.vdevice:
                self.vdevice.release()
        except Exception as e:
            logger.error(f"Error closing detector: {e}")


class MockDetector:
    def __init__(self):
        logger.info("Using mock detector for testing")
    
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        return []
    
    def close(self):
        pass


def create_detector(model_path: str = None, use_mock: bool = False) -> Optional[HailoDetector]:
    if use_mock:
        return MockDetector()
    
    try:
        return HailoDetector(model_path)
    except Exception as e:
        logger.warning(f"Failed to create Hailo detector: {e}. Using mock detector.")
        return MockDetector()
