# video_agent_gemini.py
# Video Agent with Gemini API integration - Updated for HTTP microservices
# -------------------------------------------------------------------
# Features:
# - Google Gemini core for video analysis
# - HTTP microservices integration (YOLO, scenes, ffmpeg, whisper, OCR)
# - Answer modes: json | markdown | bullets | just_number
# - Confidence scoring and provenance tracking
# - Contact sheet generation
# - Frame caching for consistency
# - Follow-up questions for ambiguity resolution
# - Verify pass for low-confidence answers

import os
import re
import gc
import io
import json
import math
import time
import base64
import requests
import logging
from dataclasses import dataclass, is_dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from collections.abc import Mapping

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Google Gemini API
import google.generativeai as genai

# LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Optional OpenCV for better keyframe extraction
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# HTTP Tool Manager - clean microservice integration
from http_tool_manager import HTTPToolManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-agent-gemini")


def _as_mapping(x):
    if is_dataclass(x) and not isinstance(x, type):
        return asdict(x)
    return x


# ===========================
# Configuration Classes
# ===========================

@dataclass
class ModelConfig:
    gemini_model: str = "gemini-1.5-flash"  # or "gemini-1.5-pro" for better quality
    api_key: Optional[str] = None  # Will be read from environment
    max_frames: int = 32
    max_output_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40

@dataclass
class AgentConfig:
    enable_reasoning: bool = True
    enable_memory: bool = True
    enable_http_services: bool = True
    max_retries: int = 2
    confidence_threshold: float = 0.5
    debug_mode: bool = False

class AnswerFormat(Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    BULLETS = "bullets"
    JUST_NUMBER = "just_number"

class ToolType(Enum):
    VISUAL_ANALYSIS = "visual_analysis"
    METADATA = "metadata_extraction"
    TIMELINE = "step_by_step"
    COUNT = "count"
    ANOMALY = "anomaly_detection"
    ASR = "asr"
    OCR = "ocr"
    SCENES = "scenes"
    RESPOND = "respond"

@dataclass
class AgentState:
    video_path: Optional[str] = None
    question: str = ""
    answer_format: AnswerFormat = AnswerFormat.MARKDOWN
    tool_type: Optional[ToolType] = None
    frames_b64: List[str] = field(default_factory=list)  # Base64 encoded frames for serialization
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    context: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[BaseMessage] = field(default_factory=list)
    
    # Helper methods for frame conversion
    def set_frames(self, frames: List[Image.Image]):
        """Convert PIL Images to base64 for serialization"""
        self.frames_b64 = [self._image_to_base64(frame) for frame in frames]
    
    def get_frames(self) -> List[Image.Image]:
        """Convert base64 back to PIL Images"""
        return [self._base64_to_image(b64) for b64 in self.frames_b64]
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _base64_to_image(self, b64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        img_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(img_data))

# ===========================
# Video Processing Utilities
# ===========================

class VideoProcessor:
    """Handles video frame extraction and metadata"""
    
    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata"""
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            # Calculate duration properly
            if container.duration:
                duration_seconds = float(container.duration * av.time_base)
            else:
                duration_seconds = 0.0
            
            # Get frame rate
            if stream.average_rate:
                fps = float(stream.average_rate)
            elif hasattr(stream, 'base_rate'):
                fps = float(stream.base_rate)
            else:
                fps = 0.0
            
            metadata = {
                "duration_seconds": duration_seconds,
                "fps": fps,
                "frames": stream.frames or 0,
                "width": stream.width,
                "height": stream.height,
                "codec": stream.codec_context.name,
                "bit_rate": stream.bit_rate,
                "pixel_format": str(stream.pix_fmt) if hasattr(stream, 'pix_fmt') else None,
                "path": video_path
            }
            
            container.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    @staticmethod
    def uniform_sample_frames(video_path: str, num_frames: int = 32) -> List[Image.Image]:
        """Extract uniformly sampled frames"""
        frames = []
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = stream.frames or 0
            
            if total_frames > 0:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:
                indices = None
            
            frame_idx = 0
            target_indices = set(indices) if indices is not None else None
            
            for frame in container.decode(video=0):
                if target_indices is None or frame_idx in target_indices:
                    img = frame.to_ndarray(format="rgb24")
                    frames.append(Image.fromarray(img))
                    if len(frames) >= num_frames:
                        break
                frame_idx += 1
            
            container.close()
            
        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
        
        return frames
    
    @staticmethod
    def extract_keyframes(video_path: str, max_frames: int = 32, threshold: float = 0.3) -> List[Image.Image]:
        """Extract keyframes using scene change detection"""
        if not CV2_AVAILABLE:
            return VideoProcessor.uniform_sample_frames(video_path, num_frames=max_frames)
        
        frames = []
        try:
            container = av.open(video_path)
            prev_hist = None
            
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")
                
                # Resize for faster histogram calculation
                img_small = cv2.resize(img, (160, 90))
                
                # Calculate color histogram
                hist = cv2.calcHist([img_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                # Detect scene change
                if prev_hist is None:
                    frames.append(Image.fromarray(img))
                    prev_hist = hist
                else:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if diff > threshold:
                        frames.append(Image.fromarray(img))
                        prev_hist = hist
                
                if len(frames) >= max_frames:
                    break
            
            container.close()
            
        except Exception as e:
            logger.error(f"Error extracting keyframes: {e}")
            return VideoProcessor.uniform_sample_frames(video_path, num_frames=max_frames)
        
        return frames

# ===========================
# Router for Intent Detection
# ===========================
class IntentRouter:
    """Routes via router microservice"""
    
    def __init__(self):
        self.logger = logging.getLogger("video-agent-router")
        self.router_url = os.getenv('ROUTER_URL', 'http://router:8006')
        self.logger.info(f"🔍 IntentRouter using URL: {self.router_url}")
    
    def _extract_format(self, question: str) -> AnswerFormat:
        """Extract answer format from question"""
        q_lower = question.lower()
        
        if 'format=json' in q_lower or 'json' in q_lower:
            return AnswerFormat.JSON
        elif 'format=bullets' in q_lower or 'bullet' in q_lower:
            return AnswerFormat.BULLETS
        elif 'format=just_number' in q_lower or 'just number' in q_lower:
            return AnswerFormat.JUST_NUMBER
        else:
            return AnswerFormat.MARKDOWN

    def classify(self, question: str, has_video: bool = True) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.router_url}/classify",
                json={"question": question, "has_video": has_video},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"✅ ROUTER SUCCESS: {result}")               
                # Map to ToolType enum
                intent_mapping = {
                    "COUNT": ToolType.COUNT,
                    "TIMELINE": ToolType.TIMELINE,
                    "VISUAL_ANALYSIS": ToolType.VISUAL_ANALYSIS,
                    "OCR": ToolType.OCR,
                    "ASR": ToolType.ASR,
                    "SCENES": ToolType.SCENES,
                    "METADATA": ToolType.METADATA,
                    "RESPOND": ToolType.RESPOND
                }
                
                tool_type = intent_mapping.get(result["intent"], ToolType.VISUAL_ANALYSIS)
                
                return {
                    "tool_type": tool_type,
                    "answer_format": self._extract_format(question),
                    "confidence": result["confidence"]
                }
        except:
            pass
        
        # Fallback to simple classification
        self.logger.info("🔄 Using fallback classification")
        return {"tool_type": ToolType.VISUAL_ANALYSIS, "answer_format": AnswerFormat.MARKDOWN, "confidence": 0.5}

# ===========================
# Gemini Core
# ===========================

class GeminiCore:
    """Core Gemini model interface for video analysis"""
    
    def __init__(self, config: ModelConfig, http_manager: HTTPToolManager):
        self.config = config
        self.http_manager = http_manager
        self.model = None
        self._frame_cache = {}
        self._load_model()
    
    def _load_model(self):
        """Load Gemini model"""
        try:
            # Configure API key
            api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
                return
            
            genai.configure(api_key=api_key)
            
            # Initialize the model
            generation_config = genai.GenerationConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=self.config.max_output_tokens,
            )
            
            self.model = genai.GenerativeModel(
                model_name=self.config.gemini_model,
                generation_config=generation_config
            )
            
            logger.info(f"Gemini model {self.config.gemini_model} loaded")
            
        except Exception as e:
            logger.error(f"Failed to load Gemini model: {e}")
    
    def get_cached_frames(self, video_path: str, use_keyframes: bool = True) -> List[Image.Image]:
        """Get frames with caching"""
        cache_key = f"{video_path}_{use_keyframes}"
        
        if cache_key not in self._frame_cache:
            if use_keyframes:
                frames = VideoProcessor.extract_keyframes(video_path, max_frames=self.config.max_frames)
            else:
                frames = VideoProcessor.uniform_sample_frames(video_path, num_frames=self.config.max_frames)
            self._frame_cache[cache_key] = frames
        
        return self._frame_cache[cache_key]
    
    def build_prompt(self, question: str, instruction: str = "") -> str:
        """Build prompt for Gemini"""
        system = "You are a helpful assistant that analyzes videos and answers questions accurately."
        
        if instruction:
            prompt = f"{system}\n\n{instruction}\n\nQuestion: {question}"
        else:
            prompt = f"{system}\n\nQuestion: {question}"
        
        return prompt
    
    def generate(self, frames: List[Image.Image], question: str, 
                 instruction: str = "", mode: str = "default") -> str:
        """Generate response using Gemini"""
        if not self.model:
            return "Model not loaded. Please set your GEMINI_API_KEY."
        
        prompt = self.build_prompt(question, instruction)
        
        try:
            # Prepare content for Gemini
            # Gemini accepts images directly
            content = [prompt]
            
            # Add frames to the content
            for i, frame in enumerate(frames[:self.config.max_frames]):
                content.append(frame)
                if i < len(frames) - 1:
                    content.append(f"Frame {i+1}:")
            
            # Generate response
            response = self.model.generate_content(content)
            
            if response.text:
                return response.text.strip()
            else:
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def extract_count(self, text: str) -> str:
        """Extract numeric count from text"""
        # Try to find a number
        match = re.search(r'\b(\d+)\b', text)
        if match:
            return match.group(1)
        
        # Try to find a range
        match = re.search(r'\b(\d+)\s*[-–]\s*(\d+)\b', text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        # Try word numbers
        word_to_num = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        
        for word, num in word_to_num.items():
            if word in text.lower():
                return num
        
        return "0"
    
    def provenance(self, frames: List[Image.Image], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate provenance information for frames used"""
        fps = metadata.get("fps", 0)
        total = metadata.get("frames", 0)
        duration = metadata.get("duration_seconds", 0)
        n_frames = len(frames)
        
        timestamps = []
        for i in range(n_frames):
            t = (duration * i) / max(n_frames - 1, 1) if duration > 0 else 0.0
            timestamps.append(round(t, 2))
        
        return {
            "frame_count_used": n_frames,
            "approx_timestamps_s": timestamps,
            "total_video_frames": total,
            "video_duration": duration
        }

# ===========================
# Contact Sheet Generator
# ===========================

def create_contact_sheet(frames: List[Image.Image], cols: int = 4) -> Image.Image:
    """Create a contact sheet from frames"""
    if not frames:
        return Image.new("RGB", (400, 300), (50, 50, 50))
    
    # Calculate dimensions
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    rows = math.ceil(len(frames) / cols)
    
    # Create sheet
    padding = 4
    sheet_w = cols * w + (cols + 1) * padding
    sheet_h = rows * h + (rows + 1) * padding
    
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    
    # Paste frames
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        x = padding + col * (w + padding)
        y = padding + row * (h + padding)
        
        # Resize frame if needed
        if frame.width != w or frame.height != h:
            frame = frame.resize((w, h), Image.Resampling.LANCZOS)
        
        sheet.paste(frame, (x, y))
    
    return sheet

# ===========================
# Agent Node Functions
# ===========================

class VideoAgent:
    """Main video analysis agent using Gemini with HTTP microservices"""
    
    def __init__(self, model_config: ModelConfig, agent_config: AgentConfig):
        self.model_config = model_config
        self.agent_config = agent_config
        
        # Use HTTP Tool Manager for microservices
        self.http_manager = HTTPToolManager()
        self.router = IntentRouter()
        self.core = GeminiCore(model_config, self.http_manager)
        
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route", self.route_node)
        workflow.add_node("extract_frames", self.extract_frames_node)
        workflow.add_node("analyze_visual", self.analyze_visual_node)
        workflow.add_node("analyze_metadata", self.analyze_metadata_node)
        workflow.add_node("analyze_timeline", self.analyze_timeline_node)
        workflow.add_node("analyze_count", self.analyze_count_node)
        workflow.add_node("analyze_asr", self.analyze_asr_node)
        workflow.add_node("analyze_ocr", self.analyze_ocr_node)
        workflow.add_node("analyze_scenes", self.analyze_scenes_node)
        workflow.add_node("respond", self.respond_node)
        workflow.add_node("verify", self.verify_node)
        workflow.add_node("format_output", self.format_output_node)
        
        # Add edges
        workflow.add_edge(START, "route")
        
        # Routing edges
        workflow.add_conditional_edges(
            "route",
            self.route_decision,
            {
                "extract_frames": "extract_frames",
                "metadata": "analyze_metadata",
                "respond": "respond"
            }
        )
        
        # Frame extraction leads to analysis
        workflow.add_conditional_edges(
            "extract_frames",
            self.analysis_decision,
            {
                "visual": "analyze_visual",
                "timeline": "analyze_timeline",
                "count": "analyze_count",
                "asr": "analyze_asr",
                "ocr": "analyze_ocr",
                "scenes": "analyze_scenes"
            }
        )
        
        # All analysis nodes go to verify
        workflow.add_edge("analyze_visual", "verify")
        workflow.add_edge("analyze_timeline", "verify")
        workflow.add_edge("analyze_count", "verify")
        workflow.add_edge("analyze_asr", "verify")
        workflow.add_edge("analyze_ocr", "verify")
        workflow.add_edge("analyze_scenes", "verify")
        workflow.add_edge("analyze_metadata", "verify")
        workflow.add_edge("respond", "verify")
        
        # Verify can retry or format
        workflow.add_conditional_edges(
            "verify",
            self.verify_decision,
            {
                "retry": "extract_frames",
                "format": "format_output"
            }
        )
        
        # End
        workflow.add_edge("format_output", END)
        
        # Compile
        memory = MemorySaver() if self.agent_config.enable_memory else None
        self.graph = workflow.compile(checkpointer=memory)
    
    # Node implementations
    
    def route_node(self, state: AgentState) -> AgentState:
        """Route to appropriate analysis"""
        result = self.router.classify(state.question, has_video=bool(state.video_path))
        state.tool_type = result["tool_type"]
        state.confidence = result["confidence"]

        logger.info(f"🎯 ROUTED: '{state.question}' → {state.tool_type.value} (confidence: {state.confidence})") 
        
        # Override format if specified
        if result["answer_format"]:
            state.answer_format = result["answer_format"]
        
        return state
    
    def route_decision(self, state: AgentState) -> str:
        """Decide routing path"""
        if state.tool_type == ToolType.RESPOND or not state.video_path:
            return "respond"
        elif state.tool_type == ToolType.METADATA:
            return "metadata"
        else:
            return "extract_frames"
   
    def extract_frames_node(self, state: AgentState) -> AgentState:
        """Extract frames with service-appropriate strategies"""
        if not state.video_path:
            return state
        
        state.metadata = VideoProcessor.get_video_metadata(state.video_path)
        duration = state.metadata.get("duration_seconds", 0)
        
        # Calculate proportional frame counts based on video duration
        base_frames = max(8, min(32, int(duration / 10)))  # 1 frame per 10 seconds, capped
        
        frame_strategies = {
            ToolType.COUNT: {
                "method": "uniform",  # YOLO works better with uniform sampling
                "count": base_frames,
                "max_count": 16  # YOLO memory limit
            },
            ToolType.OCR: {
                "method": "keyframes",  # OCR benefits from scene changes
                "count": base_frames, 
                "max_count": 12
            },
            ToolType.SCENES: {
                "method": "uniform",  # Scenes needs temporal distribution
                "count": min(base_frames * 2, 24),  # More frames for scene analysis
                "max_count": 24
            },
            ToolType.TIMELINE: {
                "method": "uniform",  # Timeline needs temporal sequence
                "count": min(base_frames * 3, 48),  # Most frames for sequence
                "max_count": 48
            },
            ToolType.VISUAL_ANALYSIS: {
                "method": "keyframes",  # Gemini benefits from varied content
                "count": base_frames,
                "max_count": 20
            }
        }
        
        strategy = frame_strategies.get(state.tool_type, frame_strategies[ToolType.VISUAL_ANALYSIS])
        
        # Extract frames using appropriate strategy
        if strategy["method"] == "uniform":
            frames = VideoProcessor.uniform_sample_frames(
                state.video_path, 
                num_frames=min(strategy["count"], strategy["max_count"])
            )
        else:  # keyframes
            frames = VideoProcessor.extract_keyframes(
                state.video_path,
                max_frames=min(strategy["count"], strategy["max_count"])
            )
        
        state.set_frames(frames)
        
        # Store strategy info for debugging
        state.provenance.update({
            "frames_used": len(frames),
            "extraction_method": strategy["method"],
            "video_duration": duration,
            "strategy": strategy
        })
        
        return state

    def analysis_decision(self, state: AgentState) -> str:
        """Decide which analysis to perform"""
        tool_map = {
            ToolType.VISUAL_ANALYSIS: "visual",
            ToolType.TIMELINE: "timeline",
            ToolType.COUNT: "count",
            ToolType.ASR: "asr",
            ToolType.OCR: "ocr",
            ToolType.SCENES: "scenes",
            ToolType.ANOMALY: "visual"
        }
        return tool_map.get(state.tool_type, "visual")
    
    def analyze_visual_node(self, state: AgentState) -> AgentState:
        """Visual analysis using Gemini"""
        instruction = (
            "Analyze the video frames and answer the question. "
            "Be specific and factual. If something is not visible, say so."
        )
        
        frames = state.get_frames()
        answer = self.core.generate(
            frames,
            state.question,
            instruction=instruction
        )
        
        confidence = 0.7
        
        # Simple retry logic without frame updates for now
        if len(answer.strip()) < 3 or answer.strip().lower() in ["not visible", "unknown"]:
            confidence = 0.5
        
        state.result = {
            "answer": answer,
            "type": "visual_analysis"
        }
        state.confidence = confidence
        
        return state
    
    def analyze_metadata_node(self, state: AgentState) -> AgentState:
        """Extract and format metadata"""
        if not state.video_path:
            state.result = {"error": "No video provided"}
            state.confidence = 0
            return state
        
        meta = VideoProcessor.get_video_metadata(state.video_path)
        
        # Format duration
        dur = meta.get("duration_seconds", 0)
        minutes = int(dur // 60)
        seconds = dur % 60
        duration_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"
        
        state.result = {
            "duration": duration_str,
            "duration_seconds": round(dur, 2),
            "fps": meta.get("fps", 0),
            "resolution": f"{meta.get('width', 0)}x{meta.get('height', 0)}",
            "codec": meta.get("codec", "unknown"),
            "frames": meta.get("frames", 0)
        }
        state.confidence = 0.95
        
        return state
    
    def analyze_timeline_node(self, state: AgentState) -> AgentState:
        """Generate step-by-step timeline"""
        instruction = (
            "Create a step-by-step timeline of events in the video.\n"
            "• Use 5-8 bullet points\n"
            "• Be specific about what happens\n"
            "• Include timestamps if possible"
        )
        
        response = self.core.generate(
            state.get_frames(),
            state.question,
            instruction=instruction
        )
        
        # Parse into bullets
        bullets = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0] in '•-*' or line[0].isdigit()):
                bullets.append(line.lstrip('•-* '))
        
        if not bullets:
            bullets = [response]
        
        state.result = {
            "timeline": bullets,
            "type": "step_by_step"
        }
        state.confidence = 0.75
        
        return state
   
    def analyze_count_node(self, state: AgentState) -> AgentState:
        """Count objects using comprehensive target extraction"""
        
        # Use router service for target extraction
        try:
            response = requests.post(
                f"{self.router.router_url}/extract_target",
                json={"question": state.question},
                timeout=10
            )
            
            if response.status_code == 200:
                target_result = response.json()
                target = target_result["target"]
                extraction_confidence = target_result["confidence"]
                logger.info(f"Target extracted: {target} (confidence: {extraction_confidence})")
            else:
                # Final fallback
                target = "person"
                extraction_confidence = 0.3
                logger.warning(f"Target extraction failed, using fallback: {target}")
                
        except Exception as e:
            logger.warning(f"Target extraction service failed: {e}")
            target = "person"
            extraction_confidence = 0.3
        
        # Use YOLO service for counting
        if self.http_manager.enabled and self.http_manager.is_service_available("yolo"):
            logger.info(f"COUNTING: {target} using YOLO (extraction confidence: {extraction_confidence})")
            
            yolo_result = self.http_manager.call_tool_sync(
                "yolo", "count_objects",
                {"frames": state.frames_b64, "class_name": target}
            )
            
            if yolo_result and "count" in yolo_result and "error" not in yolo_result:
                # Combine extraction confidence with YOLO reliability
                final_confidence = min(extraction_confidence * 0.95, 0.9)
                
                state.result = {
                    "target": target,
                    "count": yolo_result["count"],
                    "method": "phi3_extraction + yolo_counting",
                    "type": "count",
                    "extraction_confidence": extraction_confidence
                }
                state.confidence = final_confidence
                return state
            else:
                logger.warning(f"YOLO counting failed for {target}: {yolo_result}")
        
        # Fallback to Gemini counting
        instruction = f"Count the number of {target} visible in these video frames. Return only a number or range like '3' or '2-3'."
        response = self.core.generate(state.get_frames(), state.question, instruction)
        count = self.core.extract_count(response)
        
        state.result = {
            "target": target,
            "count": count,
            "method": "phi3_extraction + gemini_counting",
            "type": "count",
            "extraction_confidence": extraction_confidence
        }
        state.confidence = extraction_confidence * 0.6  # Lower confidence for Gemini counting
        
        return state
    
    def analyze_asr_node(self, state: AgentState) -> AgentState:
        """Audio/speech analysis using HTTP Whisper service"""
        # Try HTTP Whisper service
        if self.http_manager.enabled and self.http_manager.is_service_available("whisper"):
            asr_result = self.http_manager.call_tool_sync(
                "whisper", "transcribe_video",
                {"video_path": state.video_path}
            )
            
            if asr_result and "error" not in asr_result:
                state.result = {
                    "transcript": asr_result.get("text", ""),
                    "segments": asr_result.get("segments", []),
                    "method": "openai_whisper",
                    "type": "asr"
                }
                state.confidence = 0.9
                return state
        
        # Fallback message
        state.result = {
            "message": "ASR service not available. Please ensure Whisper microservice is running.",
            "type": "asr"
        }
        state.confidence = 0.3
        
        return state
    
    def analyze_ocr_node(self, state: AgentState) -> AgentState:
        """OCR text extraction using HTTP OCR service"""
        # Try HTTP OCR service
        if self.http_manager.enabled and self.http_manager.is_service_available("ocr"):
            ocr_result = self.http_manager.call_tool_sync(
                "ocr", "extract_text",
                {"frames": state.frames_b64[:8]}
            )
            
            if ocr_result and "error" not in ocr_result:
                state.result = {
                    "text": ocr_result.get("text", ""),
                    "frame_results": ocr_result.get("frame_results", []),
                    "method": "easy_ocr",
                    "type": "ocr"
                }
                state.confidence = 0.85
                return state
        
        # Fallback to Gemini
        instruction = "Read any visible text in these frames and list them."
        
        response = self.core.generate(
            state.get_frames(),
            state.question,
            instruction=instruction
        )
        
        state.result = {
            "text": response,
            "method": "gemini_fallback",
            "type": "ocr"
        }
        state.confidence = 0.6
        
        return state
    
    def analyze_scenes_node(self, state: AgentState) -> AgentState:
        """Scene detection and analysis using HTTP Scenes service"""
        # Try HTTP scene detection service
        if self.http_manager.enabled and self.http_manager.is_service_available("scenes"):
            scenes_result = self.http_manager.call_tool_sync(
                "scenes", "detect_scenes",
                {"frames": state.frames_b64}
            )
            
            if scenes_result and "error" not in scenes_result:
                state.result = {
                    "scenes": scenes_result.get("scenes", []),
                    "total_scenes": scenes_result.get("total_scenes", 0),
                    "method": "scenes_http",
                    "type": "scenes"
                }
                state.confidence = 0.85
                return state
        
        # Fallback to frame-based scene description
        instruction = "Identify distinct scenes or shots in these video frames."
        
        response = self.core.generate(
            state.get_frames(),
            state.question,
            instruction=instruction
        )
        
        state.result = {
            "description": response,
            "method": "gemini_fallback",
            "type": "scenes"
        }
        state.confidence = 0.65
        
        return state
    
    def respond_node(self, state: AgentState) -> AgentState:
        """General response without video"""
        state.result = {
            "message": (
                "Please upload a video to analyze. I can help with:\n"
                "• Counting objects (people, bikes, etc.)\n"
                "• Creating timelines\n"
                "• Extracting text (OCR)\n"
                "• Transcribing speech (ASR)\n"
                "• Scene detection\n"
                "• Visual Q&A\n\n"
                "Use format=json|markdown|bullets|just_number to control output."
            ),
            "type": "info"
        }
        state.confidence = 1.0
        
        return state
    
    def verify_node(self, state: AgentState) -> AgentState:
        """Verify results and decide if retry needed"""
        # Skip verification for high confidence or non-visual results
        if state.confidence >= 0.8 or state.tool_type in [ToolType.METADATA, ToolType.RESPOND]:
            return state
        
        # Check for low quality responses
        if state.result.get("answer"):
            answer = state.result["answer"]
            if len(answer) < 10 or answer.lower() in ["unknown", "not visible", "unclear"]:
                state.confidence *= 0.7
        
        return state
    
    def verify_decision(self, state: AgentState) -> str:
        """Decide whether to retry or format output"""
        # Only retry once and if confidence is very low
        retry_count = state.context.count({"action": "retry"}) if state.context else 0
        
        if state.confidence < self.agent_config.confidence_threshold and retry_count == 0:
            state.context.append({"action": "retry"})
            return "retry"
        
        return "format"
    
    def format_output_node(self, state: AgentState) -> AgentState:
        """Format final output based on answer format"""
        result = state.result
        
        # Add provenance
        result["provenance"] = state.provenance
        result["confidence"] = round(state.confidence, 2)
        
        # Format based on requested format
        if state.answer_format == AnswerFormat.JSON:
            # Already in dict format
            pass
        
        elif state.answer_format == AnswerFormat.BULLETS:
            if "timeline" in result:
                result = {
                    "bullets": result["timeline"],
                    "confidence": result["confidence"],
                    "provenance": result["provenance"]
                }
            elif "answer" in result:
                # Convert answer to bullets
                bullets = [line.strip() for line in result["answer"].split('\n') if line.strip()]
                result = {
                    "bullets": bullets,
                    "confidence": result["confidence"],
                    "provenance": result["provenance"]
                }
        
        elif state.answer_format == AnswerFormat.JUST_NUMBER:
            if "count" in result:
                result = str(result["count"])
            elif "answer" in result:
                # Try to extract number
                count = self.core.extract_count(result["answer"])
                result = count
        
        elif state.answer_format == AnswerFormat.MARKDOWN:
            if isinstance(result, dict):
                if "answer" in result:
                    result = result["answer"]
                elif "timeline" in result:
                    result = "\n".join(f"• {b}" for b in result["timeline"])
                elif "count" in result:
                    result = f"Count: {result['count']}"
                else:
                    # Convert dict to markdown
                    lines = []
                    for k, v in result.items():
                        if k not in ["provenance", "confidence", "type"]:
                            lines.append(f"**{k}**: {v}")
                    result = "\n".join(lines)
        
        state.result = result
        return state
    
    # Public interface - clean HTTP microservice integration
    
    def test_http_services(self):
        """Test connectivity to HTTP microservices"""
        status = self.http_manager.get_all_service_status()
        
        available_services = []
        for service, service_status in status.items():
            if "error" not in service_status:
                available_services.append(service)
                logger.info(f"✅ {service} service available")
            else:
                logger.warning(f"⚠️ {service} service unavailable: {service_status.get('error', 'Unknown error')}")
        
        if available_services:
            logger.info(f"HTTP services available: {', '.join(available_services)}")
        else:
            logger.warning("No HTTP services available - running in Gemini-only mode")

    def unwrap_result(self, res):
        # If top-level envelope dict has a "result" key -> use it
        if isinstance(res, Mapping):
            if "result" in res:
                return res["result"]
            # If it's already a result dict from your nodes (count/ocr/etc.)
            if "count" in res and res.get("type") == "count":
                # Return a human string or the number; pick one
                return f"Count: {res['count']}"
            if "answer" in res:
                return res["answer"]
            if "text" in res and res.get("type") in ["ocr", "asr"]:
                return res["text"]
            if "timeline" in res and res.get("type") == "step_by_step":
                return "\n".join(f"• {item}" for item in res["timeline"])
            # Fallback: stringify the dict
            return str(res)

        # Object with .result attribute
        if hasattr(res, "result"):
            return getattr(res, "result")

        # Pydantic BaseModel
        try:
            from pydantic import BaseModel
            if isinstance(res, BaseModel):
                try:
                    d = res.model_dump()
                except Exception:
                    d = res.dict()
                if "result" in d:
                    return d["result"]
                if "count" in d and d.get("type") == "count":
                    return f"Count: {d['count']}"
                return str(d)
        except Exception:
            pass

        # Dataclass object
        if is_dataclass(res) and not isinstance(res, type):
            d = asdict(res)
            if "result" in d:
                return d["result"]
            if "count" in d and d.get("type") == "count":
                return f"Count: {d['count']}"
            return str(d)

        # Fallback
        return res
    
    def process(self, video_path: Optional[str], question: str, 
                answer_format: AnswerFormat = AnswerFormat.MARKDOWN) -> Dict[str, Any]:
        """Process a video question"""
        initial_state = AgentState(
            video_path=video_path,
            question=question,
            answer_format=answer_format
        )
        
        config = {"configurable": {"thread_id": "video_analysis"}}
        
        result = self.graph.invoke(initial_state, config)
        
        logger.info(f"Graph.invoke returned type: {type(result)}")
        
        return self.unwrap_result(result)

# ===========================
# Main Entry Point  
# ===========================

def create_agent(api_key: Optional[str] = None) -> VideoAgent:
    """Create and configure the video agent with HTTP microservices"""
    
    model_config = ModelConfig(
        gemini_model="gemini-1.5-flash",  # Or "gemini-1.5-pro" for better quality
        api_key=api_key,
        max_frames=32,
        max_output_tokens=2048
    )
    
    agent_config = AgentConfig(
        enable_reasoning=True,
        enable_memory=True,
        enable_http_services=True,
        confidence_threshold=0.5
    )
    
    agent = VideoAgent(model_config, agent_config)
    
    # Test HTTP microservice connectivity
    agent.test_http_services()
    
    return agent
