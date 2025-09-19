#!/usr/bin/env python3
"""
YOLO HTTP Server - Object detection microservice
"""

import gc
import cv2
import time
import logging
import numpy as np
import onnxruntime as ort
import base64
import io
from typing import Dict, List, Any
from PIL import Image
from http_server_base import HTTPServer

class OptimizedYOLOONNX:
    """Memory-optimized YOLO using ONNX runtime"""
    
    def __init__(self, model_path='yolov10m.onnx'):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self.last_used = time.time()
        
        # COCO class names (YOLOv10 uses COCO dataset classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def load(self):
        """Lazy load the ONNX model"""
        if self.session is None:
            logging.info("Loading YOLO ONNX model...")
            try:
                providers = ['CPUExecutionProvider']
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.inter_op_num_threads = 2
                sess_options.intra_op_num_threads = 2
                
                self.session = ort.InferenceSession(
                    self.model_path, 
                    sess_options=sess_options,
                    providers=providers
                )
                
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                
                logging.info("✓ YOLO ONNX model loaded")
            except Exception as e:
                logging.error(f"Failed to load YOLO ONNX: {e}")
                self.session = "failed"
        self.last_used = time.time()
        
    def unload(self):
        """Unload model from memory"""
        if self.session and self.session != "failed":
            logging.info("Unloading YOLO ONNX model...")
            del self.session
            self.session = None
            gc.collect()
            
    def should_unload(self):
        """Check if model should be unloaded"""
        return (self.session and 
                self.session != "failed" and
                time.time() - self.last_used > 300)  # 5 minutes
    
    def preprocess_image(self, image):
        """Preprocess image for YOLO inference"""
        import cv2
        
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize to 640x640 (YOLO input size)
        input_size = (640, 640)
        image_resized = cv2.resize(image, input_size)
        
        # Normalize to 0-1 and change to CHW format
        image_data = image_resized.astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC -> CHW
        image_data = np.expand_dims(image_data, axis=0)   # Add batch dimension
        
        return image_data
    
    def postprocess_detections(self, outputs, confidence_threshold=0.5):
        """Postprocess YOLO outputs to get detections"""
        detections = []
        
        # YOLOv10 output format: [batch, detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
        if len(outputs) > 0:
            output = outputs[0]  # First output
            
            for detection in output[0]:  # Remove batch dimension
                x1, y1, x2, y2, conf, class_id = detection
                
                if conf > confidence_threshold:
                    class_id = int(class_id)
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        return detections
    
    def predict(self, image):
        """Run inference on image"""
        self.load()
        
        if self.session == "failed":
            return []
        
        try:
            input_data = self.preprocess_image(image)
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            detections = self.postprocess_detections(outputs)
            return detections
        except Exception as e:
            logging.error(f"ONNX prediction error: {e}")
            return []

class YOLOHTTPServer(HTTPServer):
    def __init__(self):
        super().__init__("yolo", 8011)
        self.yolo = OptimizedYOLOONNX()  # Create instance directly
        self.setup_tools()
    
    def setup_tools(self):
        """Setup YOLO-specific tool endpoints"""
        self.add_tool_endpoint("count_objects", self.count_objects)
        self.add_tool_endpoint("detect_objects", self.detect_objects)
        self.add_tool_endpoint("status", self.get_status)
    
    def count_objects(self, frames, class_name="person"):
        """Count objects in frames using ONNX"""
        try:
            if self.yolo.session == "failed":
                return {
                    "error": "YOLO ONNX model not available",
                    "count": 0,
                    "fallback": True
                }
            
            total_count = 0
            batch_size = 4
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                
                for frame_b64 in batch:
                    img_data = base64.b64decode(frame_b64)
                    image = Image.open(io.BytesIO(img_data))
                    
                    detections = self.yolo.predict(image)
                    
                    for detection in detections:
                        detected_class = detection['class_name']
                        
                        if class_name.lower() in ["person", "people"] and detected_class == "person":
                            total_count += 1
                        elif detected_class.lower() == class_name.lower():
                            total_count += 1
                
                del batch
                gc.collect()
            
            return {
                "count": total_count,
                "target": class_name,
                "method": "yolov10m_onnx",
                "frames_processed": len(frames),
                "memory_optimized": True
            }
            
        except Exception as e:
            self.logger.error(f"Count objects error: {e}")
            return {"error": str(e), "count": 0}
    
    def detect_objects(self, frames, confidence_threshold=0.5):
        """Detect all objects in frames using ONNX"""
        try:
            if self.yolo.session == "failed":
                return {"error": "YOLO ONNX model not available", "detections": []}
            
            detections = []
            
            for i, frame_b64 in enumerate(frames[:8]):
                img_data = base64.b64decode(frame_b64)
                image = Image.open(io.BytesIO(img_data))
                
                frame_detections = self.yolo.predict(image)
                
                filtered_objects = [
                    {
                        "class": det['class_name'],
                        "confidence": det['confidence'],
                        "bbox": det['bbox']
                    }
                    for det in frame_detections
                    if det['confidence'] > confidence_threshold
                ]
                
                detections.append({
                    "frame": i,
                    "objects": filtered_objects
                })
            
            return {
                "detections": detections,
                "method": "yolov10m_onnx",
                "confidence_threshold": confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Detect objects error: {e}")
            return {"error": str(e), "detections": []}
    
    def get_status(self):
        """Get service status"""
        return {
            "service": "yolo",
            "model_loaded": self.yolo.session and self.yolo.session != "failed",
            "model_type": "yolov10m_onnx" if self.yolo.session and self.yolo.session != "failed" else None,
            "model_file": "yolov10m.onnx",
            "backend": "onnxruntime",
            "memory_optimized": True
        }

if __name__ == "__main__":
    server = YOLOHTTPServer()
    server.run()
