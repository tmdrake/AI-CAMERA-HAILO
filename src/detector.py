import os
import sys
import numpy as np
from typing import List, Tuple, Optional
import logging
import threading
import queue
import time

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GObject
    GST_AVAILABLE = True
except ImportError:
    GST_AVAILABLE = False

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
        self.gst_pipeline = None
        self.model_path = model_path or "/usr/share/hailo-models/yolov5s_personface_h8l.hef"
        self._initialize()
    
    def _initialize(self):
        global Gst
        
        try:
            logger.info("Initializing Hailo with GStreamer...")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model not found: {self.model_path}")
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            Gst.init(None)
            
            # Create GStreamer pipeline with hailonet
            pipeline_str = (
                f"appsrc name=src ! "
                f"video/x-raw,format=RGB,width=640,height=640,framerate=30/1 ! "
                f"queue ! "
                f"hailonet name=hailonet hef-path={self.model_path} nms-score-threshold=0.5 force-writable=true ! "
                f"queue ! "
                f"appsink name=sink emit-signals=true"
            )
            
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsrc = self.pipeline.get_by_name("src")
            self.appsink = self.pipeline.get_by_name("sink")
            self.hailonet = self.pipeline.get_by_name("hailonet")
            
            # Set up appsink to emit new-sample signals
            self.appsink.connect("new-sample", self._on_new_sample)
            self.result_queue = queue.Queue()
            self.detection_results = []
            self.lock = threading.Lock()
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to set pipeline to PLAYING state")
            
            logger.info(f"Loaded model from: {self.model_path}")
                    
        except ImportError as e:
            logger.error(f"GStreamer not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                data = np.frombuffer(map_info.data, dtype=np.uint8)
                # Log the raw data size
                logger.info(f"Received result buffer: {len(data)} bytes")
                # Parse NMS output
                self.result_queue.put(data)
                buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        if self.pipeline is None:
            logger.error("Pipeline not initialized")
            return []
        
        try:
            import cv2
            
            logger.info(f"Processing frame: {frame.shape}")
            
            # Resize frame to model input size and convert BGR to RGB
            resized = cv2.resize(frame, (640, 640))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Create GStreamer buffer
            data = resized.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            
            # Set presentation time
            timestamp = int(time.time() * Gst.SECOND)
            buf.pts = timestamp
            buf.duration = Gst.SECOND // 30
            
            # Push frame to pipeline
            ret = self.appsrc.emit("push-buffer", buf)
            
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"push-buffer returned: {ret}")
            
            # Wait for inference to complete
            time.sleep(0.1)
            
            # Get results
            detections = []
            queue_size = self.result_queue.qsize()
            logger.info(f"Result queue size: {queue_size}")
            try:
                while True:
                    result_data = self.result_queue.get_nowait()
                    detections = self._parse_results(result_data, confidence_threshold, frame.shape)
            except queue.Empty:
                pass
            
            with self.lock:
                self.detection_results = detections
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

    def _parse_results(self, data: np.ndarray, confidence_threshold: float, frame_shape) -> List[Detection]:
        detections = []
        
        try:
            # Parse the NMS output from hailonet
            # Format for yolov5 with NMS: typically contains flattened detection data
            if len(data) == 0:
                return detections
                
            logger.info(f"Raw result size: {len(data)}")
            
            # The output is likely in NMS format
            # For yolov5 personface: 2 classes (person, face)
            # Output format: [num_detections, class_id, score, x1, y1, x2, y2]
            
            # Try parsing as float32 instead of uint8
            data_float = data.view(np.float32)
            logger.info(f"Float data sample: {data_float[:10]}")
            
            # Check for detection count at the beginning
            num_dets = int(data_float[0])
            logger.info(f"Number of detections: {num_dets}")
            
            for i in range(min(num_dets, 10)):  # Limit to 10 detections
                offset = 1 + i * 7
                if offset + 6 < len(data_float):
                    class_id = int(data_float[offset])
                    confidence = float(data_float[offset + 1])
                    x1 = float(data_float[offset + 2])
                    y1 = float(data_float[offset + 3])
                    x2 = float(data_float[offset + 4])
                    y2 = float(data_float[offset + 5])
                    
                    if class_id == self.PERSON_CLASS_ID and confidence >= confidence_threshold:
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        detections.append(Detection(
                            class_id=class_id,
                            class_name='person',
                            confidence=confidence,
                            bbox=bbox
                        ))
                        logger.info(f"Detection: class={class_id}, conf={confidence:.2f}, bbox=({x1},{y1},{x2},{y2})")
            
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
        
        return detections

    def close(self):
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
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
