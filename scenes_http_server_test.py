#!/usr/bin/env python3
"""
Scenes HTTP Server - Scene detection microservice
"""

import base64
import io
import logging
import numpy as np
import cv2
from typing import Dict, List, Any
from PIL import Image
from http_server_base import HTTPServer

class LightweightSceneDetector:
    """Fast scene detection using histogram and edge analysis"""
    
    @staticmethod
    def calculate_histogram_diff(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate histogram difference between frames"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return 1.0 - correlation
    
    @staticmethod
    def calculate_edge_diff(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate edge difference for motion detection"""
        small1 = cv2.resize(img1, (160, 120))
        small2 = cv2.resize(img2, (160, 120))
        
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        diff = cv2.absdiff(edges1, edges2)
        return np.mean(diff) / 255.0
    
    @staticmethod
    def calculate_color_diff(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate color distribution difference"""
        small1 = cv2.resize(img1, (80, 60))
        small2 = cv2.resize(img2, (80, 60))
        
        mean1 = np.mean(small1, axis=(0, 1))
        mean2 = np.mean(small2, axis=(0, 1))
        
        diff = np.linalg.norm(mean1 - mean2) / 255.0
        return min(diff, 1.0)
    
    @staticmethod
    def detect_scenes(frames: List[np.ndarray], threshold: float = 0.3) -> List[Dict]:
        """Detect scene changes using multiple lightweight metrics"""
        if len(frames) < 2:
            return [{'frame': 0, 'type': 'start', 'confidence': 1.0}]
        
        scenes = [{'frame': 0, 'type': 'start', 'confidence': 1.0}]
        
        for i in range(1, len(frames)):
            hist_diff = LightweightSceneDetector.calculate_histogram_diff(
                frames[i-1], frames[i]
            )
            edge_diff = LightweightSceneDetector.calculate_edge_diff(
                frames[i-1], frames[i]
            )
            color_diff = LightweightSceneDetector.calculate_color_diff(
                frames[i-1], frames[i]
            )
            
            combined_diff = (hist_diff * 0.5 + edge_diff * 0.3 + color_diff * 0.2)
            
            if combined_diff > threshold:
                scene_type = 'cut' if combined_diff > 0.6 else 'transition'
                scenes.append({
                    'frame': i,
                    'type': scene_type,
                    'confidence': min(combined_diff / threshold, 1.0),
                    'metrics': {
                        'histogram': round(hist_diff, 3),
                        'edge': round(edge_diff, 3),
                        'color': round(color_diff, 3)
                    }
                })
        
        return scenes
    
    @staticmethod
    def classify_scene_content(frame: np.ndarray) -> Dict:
        """Classify scene content using simple CV techniques"""
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hue_hist)
        
        scene_type = "unknown"
        if brightness < 0.3:
            scene_type = "dark"
        elif brightness > 0.7:
            scene_type = "bright"
        elif edge_density > 0.15:
            scene_type = "detailed"
        elif edge_density < 0.05:
            scene_type = "simple"
        else:
            scene_type = "normal"
        
        return {
            'type': scene_type,
            'brightness': round(brightness, 2),
            'contrast': round(contrast, 2),
            'activity': round(edge_density, 3),
            'dominant_hue': int(dominant_hue)
        }

class ScenesHTTPServer(HTTPServer):
    def __init__(self):
        super().__init__("scenes", 8014)
        self.detector = LightweightSceneDetector()  # Create instance directly
        self.setup_tools()
    
    def setup_tools(self):
        """Setup Scenes-specific tool endpoints"""
        self.add_tool_endpoint("detect_scenes", self.detect_scenes)
        self.add_tool_endpoint("classify_content", self.classify_content)
        self.add_tool_endpoint("status", self.get_status)
    
    def detect_scenes(self, frames, sensitivity="medium"):
        """Detect scene changes with optimization"""
        try:
            thresholds = {"low": 0.5, "medium": 0.3, "high": 0.2}
            threshold = thresholds.get(sensitivity, 0.3)
            
            decoded_frames = []
            max_frames = min(len(frames), 12)
            
            for frame_b64 in frames[:max_frames]:
                img_data = base64.b64decode(frame_b64)
                img = Image.open(io.BytesIO(img_data))
                
                img_array = np.array(img)
                if img_array.shape[0] > 720:
                    scale = 720 / img_array.shape[0]
                    new_size = (int(img_array.shape[1] * scale), 720)
                    img_array = cv2.resize(img_array, new_size)
                
                decoded_frames.append(img_array)
            
            scenes = self.detector.detect_scenes(decoded_frames, threshold)
            
            scene_classifications = []
            for scene in scenes[:5]:
                frame_idx = scene['frame']
                if frame_idx < len(decoded_frames):
                    classification = self.detector.classify_scene_content(
                        decoded_frames[frame_idx]
                    )
                    scene_classifications.append({
                        'frame': frame_idx,
                        'classification': classification
                    })
            
            del decoded_frames
            
            return {
                'scenes': scenes,
                'total_scenes': len(scenes),
                'frames_analyzed': max_frames,
                'sensitivity': sensitivity,
                'classifications': scene_classifications,
                'method': 'lightweight_cv',
                'memory_optimized': True
            }
            
        except Exception as e:
            self.logger.error(f"Detect scenes error: {e}")
            return {"error": str(e), "scenes": []}
    
    def classify_content(self, frame_b64):
        """Classify scene content for a single frame"""
        try:
            img_data = base64.b64decode(frame_b64)
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            
            classification = self.detector.classify_scene_content(img_array)
            
            return {
                'classification': classification,
                'method': 'lightweight_cv'
            }
            
        except Exception as e:
            self.logger.error(f"Classify content error: {e}")
            return {"error": str(e), "classification": {}}
    
    def get_status(self):
        """Get service status"""
        return {
            "service": "scenes",
            "method": "lightweight_cv",
            "features": ["scene_detection", "content_classification"],
            "sensitivities": ["low", "medium", "high"],
            "memory_optimized": True
        }

if __name__ == "__main__":
    server = ScenesHTTPServer()
    server.run()
