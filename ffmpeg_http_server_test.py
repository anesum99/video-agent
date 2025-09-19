#!/usr/bin/env python3
"""
FFmpeg HTTP Server - Video processing microservice
"""

import os
import tempfile
from http_server_base import HTTPServer

# FFmpeg HTTP Server - standalone implementation

class FFmpegHTTPServer(HTTPServer):
    def __init__(self):
        super().__init__("ffmpeg", 8015)
        self.setup_tools()
    
    def setup_tools(self):
        """Setup FFmpeg-specific tool endpoints"""
        self.add_tool_endpoint("thumbnail_grid", self.thumbnail_grid)
        self.add_tool_endpoint("cut", self.cut)
        self.add_tool_endpoint("extract_audio", self.extract_audio)
        self.add_tool_endpoint("status", self.get_status)
    
    def thumbnail_grid(self, frames, cols=4):
        """Create a thumbnail grid/contact sheet from frames"""
        try:
            import base64
            import io
            from PIL import Image
            
            # Decode frames
            images = []
            for frame_b64 in frames:
                img_data = base64.b64decode(frame_b64)
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            
            if not images:
                return {"error": "No frames provided"}
            
            # Calculate grid dimensions
            rows = (len(images) + cols - 1) // cols
            img_width = images[0].width
            img_height = images[0].height
            
            # Create contact sheet
            padding = 4
            grid_width = cols * img_width + (cols + 1) * padding
            grid_height = rows * img_height + (rows + 1) * padding
            
            grid = Image.new('RGB', (grid_width, grid_height), (30, 30, 30))
            
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                x = padding + col * (img_width + padding)
                y = padding + row * (img_height + padding)
                grid.paste(img, (x, y))
            
            # Save to temp file
            output_path = os.path.join(tempfile.gettempdir(), f"contact_sheet_{os.getpid()}.png")
            grid.save(output_path)
            
            return {
                "image_path": output_path,
                "grid_size": f"{cols}x{rows}",
                "total_frames": len(images)
            }
            
        except Exception as e:
            self.logger.error(f"Thumbnail grid error: {e}")
            return {"error": str(e)}
    
    def cut(self, video_path, start_time, end_time, output_path=None):
        """Cut a segment from video"""
        try:
            import subprocess
            
            if not output_path:
                output_path = os.path.join(tempfile.gettempdir(), f"cut_{os.getpid()}.mp4")
            
            # FFmpeg command to cut video
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',  # Copy streams without re-encoding for speed
                output_path,
                '-y'  # Overwrite output file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            duration = end_time - start_time
            
            return {
                "output_path": output_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "success": True
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg cut error: {e.stderr}")
            return {"error": f"FFmpeg error: {e.stderr}"}
        except Exception as e:
            self.logger.error(f"Cut error: {e}")
            return {"error": str(e)}
    
    def extract_audio(self, video_path, output_path=None):
        """Extract audio from video"""
        try:
            import subprocess
            
            if not output_path:
                output_path = os.path.join(tempfile.gettempdir(), f"audio_{os.getpid()}.wav")
            
            # FFmpeg command to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-ac', '1',  # Mono
                '-ar', '16000',  # 16kHz sample rate
                '-ab', '64k',  # Bit rate
                '-f', 'wav',
                output_path,
                '-y'  # Overwrite output file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                "audio_path": output_path,
                "format": "wav",
                "codec": "pcm_s16le",
                "sample_rate": 16000,
                "success": True
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg audio extraction error: {e.stderr}")
            return {"error": f"FFmpeg error: {e.stderr}"}
        except Exception as e:
            self.logger.error(f"Extract audio error: {e}")
            return {"error": str(e)}
    
    def get_status(self):
        """Get service status"""
        try:
            import subprocess
            # Test if ffmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            ffmpeg_available = result.returncode == 0
            
            return {
                "service": "ffmpeg",
                "ffmpeg_available": ffmpeg_available,
                "tools": ["thumbnail_grid", "cut", "extract_audio"],
                "formats_supported": ["mp4", "avi", "mov", "webm", "mkv", "wav"]
            }
        except Exception:
            return {
                "service": "ffmpeg",
                "ffmpeg_available": False,
                "error": "FFmpeg not found"
            }

if __name__ == "__main__":
    server = FFmpegHTTPServer()
    server.run()
