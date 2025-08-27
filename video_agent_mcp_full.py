# video_agent_mcp_full.py
# Video Agent with full MCP integration and UI/UX upgrades
# -------------------------------------------------------------------
# Features:
# - Video-LLaVA core with proper <video> prompt
# - MCP tool integration (YOLO, scenes, ffmpeg, whisper, OCR)
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
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import torch
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Transformers
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

# LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Gradio UI
import gradio as gr

# Optional OpenCV for better keyframe extraction
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# MCP support
try:
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-agent")

# ===========================
# Configuration Classes
# ===========================

@dataclass
class ModelConfig:
    videollava_model: str = "LanguageBind/Video-LLaVA-7B-hf"
    router_model: str = "microsoft/Phi-3-mini-4k-instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    max_frames: int = 32
    max_new_tokens: int = 200
    temperature: float = 0.2
    top_p: float = 0.9

@dataclass
class AgentConfig:
    enable_reasoning: bool = True
    enable_memory: bool = True
    enable_mcp: bool = True
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
    frames: List[Image.Image] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    context: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[BaseMessage] = field(default_factory=list)

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
# MCP Tool Manager
# ===========================

class MCPToolManager:
    """Manages MCP server connections and tool calls"""
    
    def __init__(self):
        self.sessions = {}
        self.enabled = MCP_AVAILABLE
        self.tool_registry = {}
    
    async def connect_server(self, name: str, command: List[str]) -> bool:
        """Connect to an MCP server"""
        if not self.enabled:
            logger.warning("MCP not available")
            return False
        
        try:
            params = StdioServerParameters(command=command, env=None)
            session = await stdio_client(params).__aenter__()
            await session.initialize()
            self.sessions[name] = session
            
            # Register available tools
            tools = await session.list_tools()
            for tool in tools.tools:
                self.tool_registry[f"{name}.{tool.name}"] = (name, tool.name)
            
            logger.info(f"Connected to MCP server '{name}' with tools: {[t.name for t in tools.tools]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            return False
    
    async def call_tool(self, server: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        if server not in self.sessions:
            raise RuntimeError(f"MCP server '{server}' not connected")
        
        session = self.sessions[server]
        result = await session.call_tool(tool, args)
        
        if hasattr(result, 'content'):
            return json.loads(result.content) if isinstance(result.content, str) else result.content
        return {}
    
    def call_tool_sync(self, server: str, tool: str, args: Dict[str, Any], timeout: float = 60) -> Dict[str, Any]:
        """Synchronous wrapper for tool calls"""
        try:
            return asyncio.run(asyncio.wait_for(
                self.call_tool(server, tool, args), 
                timeout=timeout
            ))
        except asyncio.TimeoutError:
            logger.warning(f"MCP tool call timeout: {server}.{tool}")
            return {}
        except Exception as e:
            logger.warning(f"MCP tool call failed: {server}.{tool} - {e}")
            return {}

# ===========================
# Router for Intent Detection
# ===========================

class IntentRouter:
    """Routes questions to appropriate tools"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the router model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.router_model,
                trust_remote_code=True,
                padding_side="left"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.router_model,
                torch_dtype=self.config.torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            logger.info("Router model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load router model: {e}")
    
    def classify(self, question: str, has_video: bool = True) -> Dict[str, Any]:
        """Classify the intent and extract answer format"""
        q_lower = question.lower()
        
        # Extract answer format
        if "format=json" in q_lower:
            answer_format = AnswerFormat.JSON
        elif "format=bullets" in q_lower:
            answer_format = AnswerFormat.BULLETS
        elif "format=just_number" in q_lower:
            answer_format = AnswerFormat.JUST_NUMBER
        else:
            answer_format = AnswerFormat.MARKDOWN
        
        # Determine tool type
        if not has_video:
            tool_type = ToolType.RESPOND
        elif re.search(r'\b(how many|count|number of)\b', q_lower):
            tool_type = ToolType.COUNT
        elif re.search(r'\b(step.?by.?step|timeline|sequence|chronolog|break.*down)\b', q_lower):
            tool_type = ToolType.TIMELINE
        elif re.search(r'\b(say|saying|speech|audio|talk|transcri)\b', q_lower):
            tool_type = ToolType.ASR
        elif re.search(r'\b(read|text|sign|label|poster|ocr)\b', q_lower):
            tool_type = ToolType.OCR
        elif re.search(r'\b(scene|shot|cut)\b', q_lower):
            tool_type = ToolType.SCENES
        elif re.search(r'\b(metadata|duration|fps|resolution|codec)\b', q_lower):
            tool_type = ToolType.METADATA
        elif re.search(r'\b(unusual|strange|weird|anomal|odd)\b', q_lower):
            tool_type = ToolType.ANOMALY
        else:
            tool_type = ToolType.VISUAL_ANALYSIS
        
        return {
            "tool_type": tool_type,
            "answer_format": answer_format,
            "confidence": 0.8
        }

# ===========================
# Video-LLaVA Core
# ===========================

class VideoLLaVACore:
    """Core Video-LLaVA model interface"""
    
    def __init__(self, config: ModelConfig, mcp: MCPToolManager):
        self.config = config
        self.mcp = mcp
        self.processor = None
        self.model = None
        self._frame_cache = {}
        self._load_model()
    
    def _load_model(self):
        """Load Video-LLaVA model"""
        try:
            self.processor = VideoLlavaProcessor.from_pretrained(self.config.videollava_model)
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                self.config.videollava_model,
                torch_dtype=self.config.torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            logger.info("Video-LLaVA model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load Video-LLaVA model: {e}")
    
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
        """Build proper Video-LLaVA prompt"""
        system = "You are a helpful assistant that analyzes videos and answers questions accurately."
        
        if instruction:
            prompt = f"{system}\nUSER: <video>\n{instruction}\n{question}\nASSISTANT:"
        else:
            prompt = f"{system}\nUSER: <video>\n{question}\nASSISTANT:"
        
        return prompt
    
    def generate(self, frames: List[Image.Image], question: str, 
                 instruction: str = "", mode: str = "default") -> str:
        """Generate response using Video-LLaVA"""
        if not self.model or not self.processor:
            return "Model not loaded"
        
        prompt = self.build_prompt(question, instruction)
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            videos=[frames],
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True
        }
        
        if mode == "count":
            # Strict generation for counting
            gen_kwargs.update({
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": 20
            })
        else:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repetition_penalty": 1.05
            })
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the generated part
        if "input_ids" in inputs:
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
        else:
            generated = output_ids[0]
        
        text = self.processor.batch_decode(
            generated.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return text.strip()
    
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
    """Main video analysis agent"""
    
    def __init__(self, model_config: ModelConfig, agent_config: AgentConfig):
        self.model_config = model_config
        self.agent_config = agent_config
        
        self.mcp = MCPToolManager()
        self.router = IntentRouter(model_config)
        self.core = VideoLLaVACore(model_config, self.mcp)
        
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
        """Extract frames from video"""
        if not state.video_path:
            return state
        
        # Get metadata
        state.metadata = VideoProcessor.get_video_metadata(state.video_path)
        
        # Extract frames based on tool type
        if state.tool_type == ToolType.TIMELINE:
            # Use uniform sampling for timelines
            state.frames = VideoProcessor.uniform_sample_frames(
                state.video_path, 
                num_frames=min(48, self.model_config.max_frames * 2)
            )
        else:
            # Use keyframes for other analyses
            state.frames = self.core.get_cached_frames(state.video_path)
        
        # Generate contact sheet
        if state.frames:
            sheet = create_contact_sheet(state.frames[:16])
            sheet_path = f"contact_sheet_{int(time.time())}.png"
            sheet.save(sheet_path)
            state.provenance["contact_sheet"] = sheet_path
        
        # Calculate provenance
        fps = state.metadata.get("fps", 0)
        duration = state.metadata.get("duration_seconds", 0)
        n_frames = len(state.frames)
        
        timestamps = []
        for i in range(n_frames):
            t = (duration * i) / max(n_frames - 1, 1) if duration > 0 else 0
            timestamps.append(round(t, 2))
        
        state.provenance.update({
            "frames_used": n_frames,
            "timestamps": timestamps,
            "extraction_method": "keyframes" if state.tool_type != ToolType.TIMELINE else "uniform"
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
        """Visual analysis using VLM"""
        instruction = (
            "Analyze the video frames and answer the question. "
            "Be specific and factual. If something is not visible, say so."
        )
        
        answer = self.core.generate(
            state.frames,
            state.question,
            instruction=instruction
        )
        
        state.result = {
            "answer": answer,
            "type": "visual_analysis"
        }
        state.confidence = 0.7
        
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
            state.frames,
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
        """Count objects in video"""
        # Extract what to count
        q_lower = state.question.lower()
        target_match = re.search(r'how\s+many\s+([a-zA-Z\s]+)', q_lower)
        target = target_match.group(1).strip() if target_match else "objects"
        
        # Check for ambiguity
        if target in ["people", "person"] and "background" not in q_lower and "foreground" not in q_lower:
            state.result = {
                "follow_up": "Should I count only foreground people or include background people?",
                "type": "clarification"
            }
            state.confidence = 0.4
            return state
        
        # Try MCP YOLO first
        if self.mcp.enabled:
            yolo_result = self.mcp.call_tool_sync(
                "yolo", "count_objects",
                {"frames": [self._frame_to_base64(f) for f in state.frames[:8]], "class": target}
            )
            
            if yolo_result and "count" in yolo_result:
                state.result = {
                    "target": target,
                    "count": yolo_result["count"],
                    "method": "yolo",
                    "type": "count"
                }
                state.confidence = 0.9
                return state
        
        # Fallback to VLM
        instruction = (
            f"Count the number of {target} visible in these frames.\n"
            "Return ONLY a number or range (e.g., '3' or '2-3').\n"
            "Do not include explanations."
        )
        
        response = self.core.generate(
            state.frames,
            state.question,
            instruction=instruction,
            mode="count"
        )
        
        count = self.core.extract_count(response)
        
        state.result = {
            "target": target,
            "count": count,
            "method": "vlm",
            "type": "count"
        }
        state.confidence = 0.65
        
        return state
    
    def analyze_asr_node(self, state: AgentState) -> AgentState:
        """Audio/speech analysis"""
        # Try MCP Whisper
        if self.mcp.enabled:
            asr_result = self.mcp.call_tool_sync(
                "whisper", "transcribe",
                {"video_path": state.video_path}
            )
            
            if asr_result:
                state.result = {
                    "transcript": asr_result.get("text", ""),
                    "segments": asr_result.get("segments", []),
                    "type": "asr"
                }
                state.confidence = 0.9
                return state
        
        # Fallback message
        state.result = {
            "message": "ASR requires the Whisper MCP server. Please enable it for transcription.",
            "type": "asr"
        }
        state.confidence = 0.3
        
        return state
    
    def analyze_ocr_node(self, state: AgentState) -> AgentState:
        """OCR text extraction"""
        # Try MCP OCR
        if self.mcp.enabled:
            ocr_result = self.mcp.call_tool_sync(
                "ocr", "read",
                {"frames": [self._frame_to_base64(f) for f in state.frames[:8]]}
            )
            
            if ocr_result:
                state.result = {
                    "texts": ocr_result.get("texts", []),
                    "type": "ocr"
                }
                state.confidence = 0.85
                return state
        
        # Fallback to VLM
        instruction = "Read any visible text in these frames and list them."
        
        response = self.core.generate(
            state.frames,
            state.question,
            instruction=instruction
        )
        
        state.result = {
            "texts": [response],
            "type": "ocr"
        }
        state.confidence = 0.6
        
        return state
    
    def analyze_scenes_node(self, state: AgentState) -> AgentState:
        """Scene detection and analysis"""
        # Try MCP scene detection
        if self.mcp.enabled:
            scenes_result = self.mcp.call_tool_sync(
                "scenes", "detect",
                {"video_path": state.video_path}
            )
            
            if scenes_result:
                state.result = {
                    "scenes": scenes_result.get("scenes", []),
                    "type": "scenes"
                }
                state.confidence = 0.85
                return state
        
        # Fallback to frame-based scene description
        instruction = "Identify distinct scenes or shots in these video frames."
        
        response = self.core.generate(
            state.frames,
            state.question,
            instruction=instruction
        )
        
        state.result = {
            "description": response,
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
    
    def _frame_to_base64(self, frame: Image.Image) -> str:
        """Convert frame to base64"""
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    # Public interface
    
    async def init_mcp_servers(self, servers: List[Tuple[str, List[str]]]):
        """Initialize MCP servers"""
        for name, command in servers:
            await self.mcp.connect_server(name, command)
    
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
        
        return result.result

# ===========================
# Gradio UI
# ===========================

def create_ui(agent: VideoAgent):
    """Create Gradio interface"""
    
    with gr.Blocks(title="Video Agent", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <h2 style='text-align: center'>🎬 Video Agent with MCP</h2>
        <p style='text-align: center'>
        Powered by Video-LLaVA + MCP Tools | 
        Counts • Timelines • OCR • ASR • Scenes • Contact Sheets
        </p>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Video(label="Upload Video")
                
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What's happening in this video?",
                    lines=2
                )
                
                format_dropdown = gr.Dropdown(
                    choices=["markdown", "json", "bullets", "just_number"],
                    value="markdown",
                    label="Answer Format"
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze", variant="primary")
                    clear_btn = gr.Button("Clear")
        
        with gr.Row():
            output = gr.JSON(label="Result")
        
        with gr.Row():
            status = gr.Markdown("Ready to analyze videos")
        
        # Example questions
        gr.Examples(
            examples=[
                ["How many people are in the video?"],
                ["Break this video down step by step"],
                ["What text is visible in the video?"],
                ["What's the metadata of this video?"],
                ["Describe what's happening"],
                ["How many bikes? format=just_number"],
            ],
            inputs=question_input
        )
        
        def analyze_video(video, question, format_type):
            """Process video analysis"""
            if not video:
                return {"error": "Please upload a video"}, "No video uploaded"
            
            start_time = time.time()
            
            try:
                # Convert format string to enum
                format_map = {
                    "markdown": AnswerFormat.MARKDOWN,
                    "json": AnswerFormat.JSON,
                    "bullets": AnswerFormat.BULLETS,
                    "just_number": AnswerFormat.JUST_NUMBER
                }
                answer_format = format_map.get(format_type, AnswerFormat.MARKDOWN)
                
                # Process
                result = agent.process(video, question, answer_format)
                
                elapsed = time.time() - start_time
                confidence = result.get("confidence", 0) if isinstance(result, dict) else 0.5
                
                status_msg = f"✅ Analysis complete | Confidence: {confidence:.2f} | Time: {elapsed:.1f}s"
                
                return result, status_msg
                
            except Exception as e:
                elapsed = time.time() - start_time
                return {"error": str(e)}, f"❌ Error: {e} | Time: {elapsed:.1f}s"
        
        def clear_interface():
            """Clear all inputs and outputs"""
            return None, "", "markdown", {}, "Ready to analyze videos"
        
        # Connect events
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, question_input, format_dropdown],
            outputs=[output, status]
        )
        
        clear_btn.click(
            fn=clear_interface,
            outputs=[video_input, question_input, format_dropdown, output, status]
        )
    
    return demo

# ===========================
# Main Entry Point
# ===========================

def create_agent(high_memory: bool = True) -> VideoAgent:
    """Create and configure the video agent"""
    
    model_config = ModelConfig(
        videollava_model="LanguageBind/Video-LLaVA-7B-hf",
        router_model="microsoft/Phi-3-mini-4k-instruct",
        max_frames=32 if high_memory else 16,
        max_new_tokens=200 if high_memory else 150
    )
    
    agent_config = AgentConfig(
        enable_reasoning=True,
        enable_memory=True,
        enable_mcp=True,
        confidence_threshold=0.5
    )
    
    agent = VideoAgent(model_config, agent_config)
    
    # Initialize MCP servers if available
    if MCP_AVAILABLE:
        servers = [
            # Uncomment to enable MCP servers
            # ("yolo", ["python", "mcp_yolo_server.py"]),
            # ("scenes", ["python", "mcp_scenes_server.py"]),
            # ("ffmpeg", ["python", "mcp_ffmpeg_server.py"]),
            # ("whisper", ["python", "mcp_whisper_server.py"]),
            # ("ocr", ["python", "mcp_ocr_server.py"]),
        ]
        
        # Run async initialization
        if servers:
            asyncio.run(agent.init_mcp_servers(servers))
    
    return agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Agent")
    parser.add_argument("--low-memory", action="store_true", help="Use low memory settings")
    parser.add_argument("--share", action="store_true", help="Share Gradio app publicly")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio app")
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_agent(high_memory=not args.low_memory)
    
    # Create and launch UI
    demo = create_ui(agent)
    demo.launch(share=args.share, server_port=args.port)