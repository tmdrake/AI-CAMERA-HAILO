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
        self.infer_client = None
        self.model_path = model_path
        self._initialize()
    
    def _initialize(self):
        try:
            from hailo_sdk_client import HailoSDKClient
            
            logger.info("Initializing Hailo SDK...")
            self.infer_client = HailoSDKClient()
            
            if self.model_path and os.path.exists(self.model_path):
                self.hailo_model = self.infer_client.load_hailo_model(self.model_path)
                logger.info(f"Loaded model from: {self.model_path}")
            else:
                logger.warning("Model path not specified or file not found")
                
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
            inference_results = self.infer_client.infer(self.hailo_model, frame)
            
            detections = self._parse_results(inference_results, confidence_threshold)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

    def _parse_results(self, results, confidence_threshold: float) -> List[Detection]:
        detections = []
        
        try:
            if hasattr(results, 'items'):
                for key, value in results.items():
                    if isinstance(value, dict) and 'detections' in value:
                        for det in value['detections']:
                            class_id = det.get('class_id', 0)
                            confidence = det.get('confidence', 0.0)
                            
                            if class_id == self.PERSON_CLASS_ID and confidence >= confidence_threshold:
                                bbox = (
                                    det.get('x', 0),
                                    det.get('y', 0),
                                    det.get('width', 0),
                                    det.get('height', 0)
                                )
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
        if self.infer_client:
            try:
                self.infer_client.destroy()
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
