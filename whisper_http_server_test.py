#!/usr/bin/env python3
"""
Whisper HTTP Server - Speech transcription microservice
"""

import os
import gc
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional
from http_server_base import HTTPServer

class OptimizedWhisper:
    """Memory-optimized Whisper with auto-unloading"""
    
    def __init__(self, model_size="tiny"):
        self.model_size = model_size
        self.model = None
        self.last_used = time.time()
        
    def load(self):
        """Lazy load Whisper model"""
        if self.model is None:
            logging.info(f"Loading Whisper {self.model_size} model...")
            try:
                import whisper
                self.model = whisper.load_model(self.model_size)
                self.model.eval()
                logging.info(f"✓ Whisper {self.model_size} loaded")
            except Exception as e:
                logging.error(f"Failed to load Whisper: {e}")
                self.model = "failed"
        self.last_used = time.time()
        
    def unload(self):
        """Unload model from memory"""
        if self.model and self.model != "failed":
            logging.info("Unloading Whisper model...")
            del self.model
            self.model = None
            gc.collect()
            
    def should_unload(self):
        """Check if model should be unloaded"""
        return (self.model and 
                self.model != "failed" and
                time.time() - self.last_used > 180)
    
    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe with optimizations"""
        self.load()
        
        if self.model == "failed":
            return {"error": "Model failed to load"}
        
        try:
            result = self.model.transcribe(
                audio_path,
                fp16=False,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0,
                condition_on_previous_text=False,
                verbose=False
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return {"error": str(e)}
        finally:
            if self.should_unload():
                self.unload()

def extract_audio_optimized(video_path: str) -> Optional[str]:
    """Extract audio using FFmpeg with optimization"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            audio_path = tmp_audio.name
            
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',
            '-ac', '1',
            '-ar', '16000',
            '-ab', '32k',
            '-f', 'wav',
            audio_path,
            '-y'
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return audio_path
        
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        return None

class WhisperHTTPServer(HTTPServer):
    def __init__(self):
        super().__init__("whisper", 8012)
        self.whisper = OptimizedWhisper("tiny")  # Create instance directly
        self.setup_tools()
    
    def setup_tools(self):
        """Setup Whisper-specific tool endpoints"""
        self.add_tool_endpoint("transcribe_video", self.transcribe_video)
        self.add_tool_endpoint("transcribe_audio", self.transcribe_audio)
        self.add_tool_endpoint("status", self.get_status)
    
    def transcribe_video(self, video_path):
        """Transcribe video with memory optimization"""
        try:
            # Extract audio with optimization
            audio_path = extract_audio_optimized(video_path)
            if not audio_path:
                return {"error": "Failed to extract audio", "text": ""}
            
            try:
                # Transcribe
                result = self.whisper.transcribe(audio_path)
                
                # Clean up audio file immediately
                Path(audio_path).unlink(missing_ok=True)

                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", "en"),
                    "segments": result.get("segments", [])[:10],
                    "method": "whisper_tiny_optimized"
                }
                
            finally:
                Path(audio_path).unlink(missing_ok=True)
                gc.collect()
                
        except Exception as e:
            self.logger.error(f"Transcribe video error: {e}")
            return {"error": str(e), "text": ""}
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file directly"""
        try:
            result = self.whisper.transcribe(audio_path)
            
            return {
                "text": result.get("text", ""),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])[:20],
                "method": "whisper_tiny_optimized"
            }
            
        except Exception as e:
            self.logger.error(f"Transcribe audio error: {e}")
            return {"error": str(e), "text": ""}
    
    def get_status(self):
        """Get service status"""
        return {
            "service": "whisper",
            "model_loaded": self.whisper.model and self.whisper.model != "failed",
            "model_type": "whisper-tiny" if self.whisper.model else None,
            "memory_optimized": True
        }

if __name__ == "__main__":
    server = WhisperHTTPServer()
    server.run()
