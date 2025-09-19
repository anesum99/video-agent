#!/usr/bin/env python3
"""
OCR HTTP Server - Text recognition microservice
"""

import gc
import base64
import io
import logging
import time
import numpy as np
from typing import Dict, List, Any
from PIL import Image
from http_server_base import HTTPServer

class OptimizedOCR:
    """Memory-optimized OCR with selective loading"""
    
    def __init__(self, primary_engine="tesseract"):
        self.primary_engine = primary_engine
        self.easyocr_reader = None
        self.last_used = time.time()
        
    def load_easyocr(self):
        """Lazy load EasyOCR only when needed"""
        if self.easyocr_reader is None:
            logging.info("Loading EasyOCR model...")
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(
                    ['en'], 
                    gpu=False,
                    model_storage_directory='/tmp/easyocr',
                    download_enabled=True,
                    detector=True,
                    recognizer=True,
                    verbose=False
                )
                logging.info("✓ EasyOCR loaded")
            except Exception as e:
                logging.error(f"Failed to load EasyOCR: {e}")
                self.easyocr_reader = "failed"
        self.last_used = time.time()
        
    def unload_easyocr(self):
        """Unload EasyOCR model"""
        if self.easyocr_reader and self.easyocr_reader != "failed":
            logging.info("Unloading EasyOCR model...")
            del self.easyocr_reader
            self.easyocr_reader = None
            gc.collect()
            
    def should_unload(self):
        """Check if models should be unloaded"""
        return time.time() - self.last_used > 180  # 3 minutes
    
    def extract_with_tesseract(self, image: Image.Image) -> List[Dict]:
        """Lightweight Tesseract extraction"""
        try:
            import pytesseract
            
            gray = image.convert('L')
            data = pytesseract.image_to_data(
                gray, 
                output_type=pytesseract.Output.DICT,
                config='--psm 11'
            )
            
            results = []
            for i, text in enumerate(data['text']):
                if text.strip() and data['conf'][i] > 30:
                    results.append({
                        'text': text,
                        'confidence': data['conf'][i] / 100.0,
                        'bbox': [
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ]
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Tesseract error: {e}")
            return []
    
    def extract_with_easyocr(self, image: Image.Image) -> List[Dict]:
        """Heavy EasyOCR extraction for complex cases"""
        self.load_easyocr()
        
        if self.easyocr_reader == "failed":
            return []
        
        try:
            img_array = np.array(image)
            results = self.easyocr_reader.readtext(
                img_array,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                slope_ths=0.3,
                batch_size=1
            )
            
            formatted = []
            for bbox, text, conf in results:
                if conf > 0.3:
                    formatted.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': [int(x) for point in bbox for x in point[:2]][:4]
                    })
            
            return formatted
            
        except Exception as e:
            logging.error(f"EasyOCR error: {e}")
            return []
        finally:
            if self.should_unload():
                self.unload_easyocr()
    
    def extract_adaptive(self, image: Image.Image) -> Dict:
        """Adaptive extraction - use Tesseract first, fall back to EasyOCR"""
        tesseract_results = self.extract_with_tesseract(image)
        
        total_conf = sum(r['confidence'] for r in tesseract_results)
        avg_conf = total_conf / len(tesseract_results) if tesseract_results else 0
        
        if avg_conf > 0.7 and len(tesseract_results) > 0:
            return {
                'text': ' '.join(r['text'] for r in tesseract_results),
                'results': tesseract_results[:20],
                'method': 'tesseract',
                'confidence': avg_conf
            }
        else:
            logging.info("Tesseract confidence low, trying EasyOCR...")
            easyocr_results = self.extract_with_easyocr(image)
            
            all_results = tesseract_results + easyocr_results
            seen_texts = set()
            unique_results = []
            for r in all_results:
                if r['text'].lower() not in seen_texts:
                    unique_results.append(r)
                    seen_texts.add(r['text'].lower())
            
            return {
                'text': ' '.join(r['text'] for r in unique_results),
                'results': unique_results[:20],
                'method': 'adaptive',
                'confidence': max(avg_conf, 
                                sum(r['confidence'] for r in easyocr_results) / len(easyocr_results) if easyocr_results else 0)
            }

class OCRHTTPServer(HTTPServer):
    def __init__(self):
        super().__init__("ocr", 8013)
        self.ocr = OptimizedOCR()  # Create instance directly
        self.setup_tools()
    
    def setup_tools(self):
        """Setup OCR-specific tool endpoints"""
        self.add_tool_endpoint("extract_text", self.extract_text)
        self.add_tool_endpoint("read_frame", self.read_frame)
        self.add_tool_endpoint("status", self.get_status)
    
    def extract_text(self, frames, use_heavy=False):
        """Extract text from frames with optimization"""
        try:
            all_text = []
            frame_results = []
            
            max_frames = 4 if use_heavy else 6
            
            for i, frame_b64 in enumerate(frames[:max_frames]):
                img_data = base64.b64decode(frame_b64)
                image = Image.open(io.BytesIO(img_data))
                
                if image.width > 1280:
                    ratio = 1280 / image.width
                    new_size = (1280, int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                if use_heavy:
                    results = self.ocr.extract_with_easyocr(image)
                    method = "easyocr"
                else:
                    result_dict = self.ocr.extract_adaptive(image)
                    results = result_dict.get('results', [])
                    method = result_dict.get('method', 'adaptive')
                
                if results:
                    frame_text = ' '.join(r['text'] for r in results)
                    all_text.append(frame_text)
                    frame_results.append({
                        'frame': i,
                        'text': frame_text[:200],
                        'count': len(results)
                    })
                
                del image
                gc.collect()
            
            return {
                'text': ' '.join(all_text),
                'frames_processed': len(frames[:max_frames]),
                'frame_results': frame_results,
                'method': method,
                'optimized': True
            }
            
        except Exception as e:
            self.logger.error(f"Extract text error: {e}")
            return {"error": str(e), "text": ""}
    
    def read_frame(self, frame_b64, engine="adaptive"):
        """Read text from a single frame"""
        try:
            img_data = base64.b64decode(frame_b64)
            image = Image.open(io.BytesIO(img_data))
            
            if engine == "tesseract":
                results = self.ocr.extract_with_tesseract(image)
                method = "tesseract"
            elif engine == "easyocr":
                results = self.ocr.extract_with_easyocr(image)
                method = "easyocr"
            else:
                result_dict = self.ocr.extract_adaptive(image)
                results = result_dict.get('results', [])
                method = result_dict.get('method', 'adaptive')
            
            text = ' '.join(r['text'] for r in results) if results else ""
            
            return {
                'text': text,
                'results': results[:10],
                'method': method,
                'confidence': sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
            }
            
        except Exception as e:
            self.logger.error(f"Read frame error: {e}")
            return {"error": str(e), "text": ""}
    
    def get_status(self):
        """Get service status"""
        return {
            "service": "ocr",
            "engines": ["tesseract", "easyocr", "adaptive"],
            "easyocr_loaded": self.ocr.easyocr_reader is not None and self.ocr.easyocr_reader != "failed",
            "memory_optimized": True
        }

if __name__ == "__main__":
    server = OCRHTTPServer()
    server.run()
