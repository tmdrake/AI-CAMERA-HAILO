import os
import sys
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Detection:
    def __init__(self, class_id: int, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox

class HailoDetector:
    PERSON_CLASS_ID = 0
    
    def __init__(self, model_path: str = None):
        self.hailo_model = None
        self.infer_engine = None
        self.model_path = model_path or "/usr/share/hailo-models/yolov8n_persons/yolov8n_persons.onnx"
        self._initialize()
    
    def _initialize(self):
        try:
            import hailo
            
            logger.info("Initializing Hailo SDK...")
            
            if os.path.exists(self.model_path):
                self.hailo_model = hailo.HailoModel(self.model_path)
                logger.info(f"Loaded model from: {self.model_path}")
            else:
                logger.warning(f"Model not found at: {self.model_path}, using fallback")
                self.model_path = "/usr/share/hailo-models/yolov8n/yolov8n.ckpt"
                if os.path.exists(self.model_path):
                    self.hailo_model = hailo.HailoModel(self.model_path)
                    logger.info(f"Loaded fallback model from: {self.model_path}")
                    
        except ImportError:
            logger.error("Hailo SDK not installed. Install with: sudo apt install hailo-all")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        if self.hailo_model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            import cv2
            
            input_size = (640, 640)
            resized = cv2.resize(frame, input_size)
            input_data = resized.astype(np.float32) / 255.0
            input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            inference_results = self.hailo_model.infer(input_data)
            
            detections = self._parse_results(inference_results, confidence_threshold, frame.shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

    def _parse_results(self, results, confidence_threshold: float, frame_shape) -> List[Detection]:
        detections = []
        
        try:
            if hasattr(results, 'items'):
                for key, value in results.items():
                    if isinstance(value, np.ndarray) and len(value) > 0:
                        for det in value[0]:
                            if len(det) >= 6:
                                class_id = int(det[1])
                                confidence = float(det[2])
                                
                                if class_id == self.PERSON_CLASS_ID and confidence >= confidence_threshold:
                                    x1, y1, x2, y2 = det[3:7]
                                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                                    detections.append(Detection(
                                        class_id=class_id,
                                        class_name='person',
                                        confidence=confidence,
                                        bbox=bbox
                                    ))
            
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
        
        return detections

    def close(self):
        if self.hailo_model:
            try:
                del self.hailo_model
            except:
                pass


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
