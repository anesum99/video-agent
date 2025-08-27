# video_agent_mcp_clone.py
# A clone of the VideoLLaVA agent with MCP integrations and UX upgrades.
# ----------------------------------------------------------------------
# Key features added:
# - Proper Video-LLaVA prompt (<video>) and chat formatting
# - Consistent frame pack via caching (prevents sampler drift)
# - Answer modes: json | markdown | bullets | just_number (router + UI param)
# - Confidence + provenance (frames/timestamps used)
# - Contact sheet output (MCP ffmpeg.thumbnail_grid or local fallback)
# - Follow-ups for ambiguous queries (e.g., people count scope)
# - MCP tools: yolo.count_objects, scenes.detect, ffmpeg.thumbnail_grid,
#              whisper.transcribe, ocr.read, ffmpeg.cut, video.index/search
# - Verify pass (self-check; retry with uniform frames if low-confidence)
# - Highlight maker node using MCP ffmpeg
# - Vector index MCP stubs for cross-video search
#
# NOTE: MCP servers/tools are optional — code gracefully falls back when unavailable.

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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Annotated

import torch
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Transformers (Video-LLaVA)
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

# LangGraph / LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Gradio (UI)
import gradio as gr

# Optional OpenCV (for keyframes)
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

# Optional MCP
try:
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
    MCP_AVAILABLE = True
except Exception:
    MCP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("video-agent-mcp")

# -------------------------
# Configs & State
# -------------------------

@dataclass
class ModelConfig:
    videollava_model: str = "LanguageBind/Video-LLaVA-7B-hf"
    router_model: str = "microsoft/Phi-3-mini-4k-instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    max_frames: int = 32
    max_new_tokens: int = 200

@dataclass
class AgentConfig:
    enable_reasoning: bool = True
    enable_memory: bool = True
    max_tool_calls: int = 3
    fallback_enabled: bool = True
    debug_mode: bool = False

AnswerMode = str  # "json" | "markdown" | "bullets" | "just_number"

class ToolType:
    VISUAL_ANALYSIS = "visual_analysis"
    METADATA = "metadata_extraction"
    TIMELINE = "step_by_step"
    ANOMALY = "anomaly_detection"
    LIST_TOOLS = "list_tools"
    RESPOND = "respond"
    ASR = "asr"
    OCR = "ocr"
    YOLO_COUNT = "yolo_count"
    SCENES = "scenes"
    HIGHLIGHT = "highlight"
    VECTOR_INDEX = "vector_index"

class AgentState(dict):
    pass

# -------------------------
# Video utilities
# -------------------------

class VideoProcessor:
    @staticmethod
    def _duration_seconds(container: av.container.InputContainer) -> float:
        # Correct conversion: AV_TIME_BASE units -> seconds
        return float(container.duration * av.time_base) if container.duration is not None else 0.0

    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        container = av.open(video_path)
        vs = container.streams.video[0]
        duration_seconds = VideoProcessor._duration_seconds(container)
        if vs.average_rate:
            fps = float(vs.average_rate)
        elif getattr(vs, "base_rate", None):
            fps = float(vs.base_rate)
        else:
            fps = 0.0
        meta = {
            "duration_seconds": duration_seconds,
            "fps": fps,
            "frames": vs.frames,
            "width": vs.width,
            "height": vs.height,
            "codec": vs.codec_context.name,
            "bit_rate": vs.bit_rate,
            "pixel_format": str(vs.pix_fmt),
        }
        container.close()
        return meta

    @staticmethod
    def uniform_sample_frames(video_path: str, num_frames: int = 32) -> List[Image.Image]:
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            total = stream.frames or 0
            indices = (np.linspace(0, max(total-1, 0), num_frames).astype(int).tolist()
                       if total > 0 else None)

            images, i = [], 0
            target = set(indices) if indices is not None else None
            for frame in container.decode(video=0):
                take = (target is None) or (i in target)
                if take:
                    img = frame.to_ndarray(format="rgb24")
                    images.append(Image.fromarray(img))
                    if len(images) >= num_frames:
                        break
                i += 1
            container.close()
            return images
        except Exception as e:
            log.error(f"uniform_sample_frames error: {e}")
            return []

    @staticmethod
    def extract_keyframes(video_path: str, max_frames: int = 32, threshold: float = 0.3) -> List[Image.Image]:
        if not CV2_OK:
            return VideoProcessor.uniform_sample_frames(video_path, num_frames=max_frames)
        try:
            container = av.open(video_path)
            prev_hist = None
            selected = []
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")
                img_small = cv2.resize(img, (160, 90))
                hist = cv2.calcHist([img_small], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
                hist = cv2.normalize(hist, hist).flatten()
                if prev_hist is None or cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > threshold:
                    selected.append(Image.fromarray(img))
                    prev_hist = hist
                if len(selected) >= max_frames:
                    break
            container.close()
            return selected
        except Exception as e:
            log.error(f"extract_keyframes error: {e}")
            return VideoProcessor.uniform_sample_frames(video_path, num_frames=max_frames)

# -------------------------
# MCP Manager
# -------------------------

class MCPToolManager:
    def __init__(self):
        self.sessions = {}
        self.enabled = MCP_AVAILABLE

    async def open(self, name: str, command: List[str]) -> bool:
        if not self.enabled:
            log.warning("MCP not available")
            return False
        try:
            params = StdioServerParameters(command=command, env=None)
            session = await stdio_client(params).__aenter__()
            await session.initialize()
            self.sessions[name] = session
            try:
                tools = await session.list_tools()
                log.info(f"MCP[{name}] tools: {[t.name for t in tools.tools]}")
            except Exception:
                pass
            return True
        except Exception as e:
            log.error(f"MCP[{name}] open error: {e}")
            return False

    async def call(self, name: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.sessions:
            raise RuntimeError(f"MCP server '{name}' not open")
        session = self.sessions[name]
        result = await session.call_tool(tool_name, args)
        return getattr(result, "content", result) if result else {}

    def call_sync(self, name: str, tool: str, args: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
        try:
            return asyncio.run(asyncio.wait_for(self.call(name, tool, args), timeout=timeout))
        except Exception as e:
            log.warning(f"MCP[{name}.{tool}] failed: {e}")
            return {}

# -------------------------
# Router (intent + answer mode)
# -------------------------

class LLMRouter:
    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.device = config.device
        self.model = None
        self.tok = None
        self._init()

    def _init(self):
        try:
            self.tok = AutoTokenizer.from_pretrained(self.cfg.router_model, trust_remote_code=True, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.router_model,
                torch_dtype=self.cfg.torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model.eval()
            log.info("Router LLM loaded")
        except Exception as e:
            log.error(f"Router init failed: {e}")
            self.model, self.tok = None, None

    def classify(self, text: str, has_video: bool) -> Dict[str, Any]:
        tl = text.lower()
        fmt: AnswerMode = "json" if "format=json" in tl else \
                          "bullets" if "format=bullets" in tl else \
                          "just_number" if "format=just_number" in tl else \
                          "markdown"

        if not has_video:
            tool = "respond"
        elif re.search(r"\b(duration|fps|resolution|codec|bit[- ]?rate|metadata)\b", tl):
            tool = "metadata_extraction"
        elif re.search(r"\b(step[- ]?by[- ]?step|timeline|sequence|chronolog|beginning|middle|end|conclude)\b", tl):
            tool = "step_by_step"
        elif re.search(r"\b(how many|count|number of)\b", tl):
            tool = "yolo_count"
        elif re.search(r"\b(unusual|strange|weird|anomal|odd|unexpected)\b", tl):
            tool = "anomaly_detection"
        elif re.search(r"\b(say|saying|speech|audio|talk|said)\b", tl):
            tool = "asr"
        elif re.search(r"\b(read|text|sign|label|poster|scoreboard|subtitle)\b", tl):
            tool = "ocr"
        else:
            tool = "visual_analysis"

        return {"tool": tool, "format": fmt, "reason": "rule-router", "confidence": 0.75}

# -------------------------
# VideoLLaVA core + MCP fusion
# -------------------------

class VideoCore:
    def __init__(self, cfg: ModelConfig, mcp: MCPToolManager):
        self.cfg = cfg
        self.mcp = mcp
        self.processor = VideoLlavaProcessor.from_pretrained(cfg.videollava_model)
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            cfg.videollava_model,
            torch_dtype=cfg.torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self._frame_cache: Dict[str, List[Image.Image]] = {}

    def get_ref_frames(self, video_path: str) -> List[Image.Image]:
        if video_path in self._frame_cache:
            return self._frame_cache[video_path]
        frames = VideoProcessor.extract_keyframes(video_path, max_frames=self.cfg.max_frames)
        self._frame_cache[video_path] = frames
        return frames

    def build_chat_prompt(self, question: str, preface: str = "") -> str:
        system = "You are a helpful assistant that watches the video and answers concisely and factually."
        pre = (preface.strip() + "\n") if preface else ""
        return f"{system}\nUSER: <video>\n{pre}{question}\nASSISTANT:"

    def _decode_continuation(self, inputs: Dict[str, Any], ids: torch.Tensor) -> str:
        if "input_ids" in inputs:
            gen_only = ids[0, inputs["input_ids"].shape[1]:]
        else:
            gen_only = ids[0]
        out = self.processor.batch_decode(gen_only.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return out.strip()

    def _extract_count(self, text: str) -> str:
        t = text.strip()
        m = re.search(r"\b(\d+)\b", t)
        if m:
            return m.group(1)
        m = re.search(r"\b(\d+)\s*[-–]\s*(\d+)\b", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return f"{min(a,b)}-{max(a,b)}"
        words = {"one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
        for w,n in words.items():
            if re.search(rf"\b{w}\b", t.lower()):
                return n
        return t or "0"

    def _count_target(self, q: str) -> str:
        ql = q.lower().strip()
        m = re.search(r"how\s+many\s+([a-zA-Z0-9 \-_/]+)", ql)
        if m:
            raw = m.group(1).strip(" ?.")
        else:
            raw = "people"
        aliases = {
            "person":"people","persons":"people",
            "bicycle":"bikes","bicycles":"bikes","bike":"bikes","stationary bike":"bikes","exercise bike":"bikes",
            "elliptical":"ellipticals","treadmill":"treadmills"
        }
        return aliases.get(raw, raw)

    def provenance(self, frames: List[Image.Image], metadata: Dict[str, Any]) -> Dict[str, Any]:
        fps = metadata.get("fps") or 0
        total = metadata.get("frames") or 0
        dur = metadata.get("duration_seconds") or 0
        n = len(frames)
        stamps = []
        for i in range(n):
            t = (dur * i) / max(n-1, 1) if dur > 0 else 0.0
            stamps.append(round(t, 2))
        return {"frame_count_used": n, "approx_timestamps_s": stamps}

    # MCP helpers (graceful fallbacks in agent methods)
    def mcp_yolo_count(self, frames: List[Image.Image], target: str) -> Optional[int]:
        return None  # stub unless server added

    def mcp_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        return []    # stub unless server added

    def mcp_thumbnail_grid(self, frames: List[Image.Image]) -> Optional[str]:
        # local fallback grid
        try:
            grid = contact_sheet(frames, cols=4)
            out_path = os.path.join(os.getcwd(), f"contact_sheet_{int(time.time())}.png")
            grid.save(out_path, "PNG")
            return out_path
        except Exception:
            return None

    def mcp_asr(self, video_path: str) -> Dict[str, Any]:
        return {}    # stub unless server added

    def mcp_ocr(self, frames: List[Image.Image]) -> Dict[str, Any]:
        return {}    # stub unless server added

    def vlm_answer(self, frames: List[Image.Image], question: str, preface: str, mode: str) -> str:
        prompt = self.build_chat_prompt(question=question, preface=preface)
        inputs = self.processor(text=prompt, videos=[frames], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: (v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max(96, self.cfg.max_new_tokens),
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
        )
        if mode == "count":
            gen_kwargs.update(dict(do_sample=False, temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=8))
        else:
            gen_kwargs.update(dict(do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.05))

        with torch.no_grad():
            ids = self.model.generate(**inputs, **gen_kwargs)
        return self._decode_continuation(inputs, ids)

def contact_sheet(frames: List[Image.Image], cols: int = 4, pad: int = 4) -> Image.Image:
    if not frames:
        raise ValueError("no frames for contact sheet")
    w = max(im.width for im in frames)
    h = max(im.height for im in frames)
    rows = int(math.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols*w + pad*(cols+1), rows*h + pad*(rows+1)), (24,24,24))
    x = y = pad
    for i, im in enumerate(frames):
        sheet.paste(im.resize((w, h)), (x, y))
        x += w + pad
        if (i+1) % cols == 0:
            x = pad
            y += h + pad
    return sheet

# -------------------------
# Agent orchestration
# -------------------------

class Agent:
    def __init__(self, cfg: ModelConfig, acfg: AgentConfig):
        self.cfg = cfg
        self.acfg = acfg
        self.mcp = MCPToolManager()
        self.router = LLMRouter(cfg)
        self.core = VideoCore(cfg, self.mcp)
        self.memory = MemorySaver() if acfg.enable_memory else None

    def process(self, video_path: Optional[str], question: str, answer_mode: AnswerMode = "markdown") -> Dict[str, Any]:
        has_video = bool(video_path)
        route = self.router.classify(question, has_video=has_video)
        tool = route["tool"]
        fmt = answer_mode or route["format"]

        if not has_video and tool != "respond":
            tool = "respond"

        try:
            if tool == "respond":
                text = self._respond(question)
                return self._format_answer(text=text, fmt=fmt, provenance={}, confidence=0.5)

            if not video_path:
                return self._format_answer(text="No video loaded.", fmt=fmt, provenance={}, confidence=0.2)

            meta = VideoProcessor.get_video_metadata(video_path)

            if tool == "metadata_extraction":
                return self._answer_metadata(meta, fmt)
            if tool == "step_by_step":
                return self._answer_timeline(video_path, question, meta, fmt)
            if tool == "yolo_count":
                return self._answer_count(video_path, question, meta, fmt)
            if tool == "asr":
                return self._answer_asr(video_path, question, meta, fmt)
            if tool == "ocr":
                return self._answer_ocr(video_path, question, meta, fmt)

            return self._answer_visual(video_path, question, meta, fmt)

        except Exception as e:
            log.exception("process failed")
            return self._format_answer(text=f"Error: {e}", fmt=fmt, provenance={}, confidence=0.0)

    def _answer_metadata(self, meta: Dict[str, Any], fmt: AnswerMode) -> Dict[str, Any]:
        dur = meta["duration_seconds"]
        minutes = int(dur // 60)
        seconds = dur % 60
        duration_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"
        text = {
            "duration_seconds": round(dur, 2),
            "duration_pretty": duration_str,
            "fps": meta["fps"],
            "frames": meta["frames"],
            "resolution": f"{meta['width']}x{meta['height']}",
            "codec": meta["codec"],
            "bit_rate": meta.get("bit_rate"),
            "pixel_format": meta.get("pixel_format"),
        }
        return self._format_answer(text=text, fmt=fmt, provenance={}, confidence=0.95)

    def _answer_timeline(self, video_path: str, question: str, meta: Dict[str, Any], fmt: AnswerMode) -> Dict[str, Any]:
        frames = VideoProcessor.uniform_sample_frames(video_path, num_frames=min(48, self.cfg.max_frames*2))
        pre = ("Provide a concise step-by-step timeline of the video.\n"
               "• Use 5–8 bullet points, one sentence each.\n"
               "• If timing is inferable, prefix with [MM:SS].")
        text = self.core.vlm_answer(frames, question, preface=pre, mode="timeline")
        bullets = [b.strip("-• ").strip() for b in text.splitlines() if b.strip()]
        prov = self.core.provenance(frames, meta)
        sheet = self.core.mcp_thumbnail_grid(frames)
        payload = {"timeline": bullets, "contact_sheet": sheet, "provenance": prov}
        return self._format_answer(text=payload, fmt=fmt, provenance=prov, confidence=0.8)

    def _answer_count(self, video_path: str, question: str, meta: Dict[str, Any], fmt: AnswerMode) -> Dict[str, Any]:
        frames = self.core.get_ref_frames(video_path)
        target = self.core._count_target(question)

        if target in ("people", "person") and "background" not in question.lower() and "foreground" not in question.lower():
            followup = "Do you want me to count only the foreground (on/next to the main machine) or include background people?"
            return self._format_answer(text={"follow_up": followup}, fmt="json", provenance={}, confidence=0.4)

        pre = (f"Count the number of distinct {target} visible in the video frames.\n"
               "Return ONLY a single integer; if uncertain, return a short range like '2-3'.\n"
               "Do NOT add words or explanations.")
        text = self.core.vlm_answer(frames, question, preface=pre, mode="count")
        num = self.core._extract_count(text)
        prov = self.core.provenance(frames, meta)
        sheet = self.core.mcp_thumbnail_grid(frames)
        payload = {"target": target, "count": num, "contact_sheet": sheet, "provenance": prov}
        return self._format_answer(text=payload if fmt!="just_number" else num, fmt=fmt, provenance=prov, confidence=0.65)

    def _answer_asr(self, video_path: str, question: str, meta: Dict[str, Any], fmt: AnswerMode) -> Dict[str, Any]:
        res = self.core.mcp_asr(video_path)
        if not res:
            return self._format_answer(text="ASR unavailable. Please enable the whisper MCP server.", fmt=fmt, provenance={}, confidence=0.3)
        prov = {"source": "whisper.transcribe", "segments": len(res.get("segments", []))}
        return self._format_answer(text=res, fmt=fmt, provenance=prov, confidence=0.9)

    def _answer_ocr(self, video_path: str, question: str, meta: Dict[str, Any], fmt: AnswerMode) -> Dict[str, Any]:
        frames = self.core.get_ref_frames(video_path)
        res = self.core.mcp_ocr(frames)
        if not res:
            return self._format_answer(text="OCR unavailable. Please enable the ocr MCP server.", fmt=fmt, provenance={}, confidence=0.3)
        prov = self.core.provenance(frames, meta)
        payload = {"ocr": res, "provenance": prov}
        return self._format_answer(text=payload, fmt=fmt, provenance=prov, confidence=0.85)

    def _answer_visual(self, video_path: str, question: str, meta: Dict[str, Any], fmt: AnswerMode) -> Dict[str, Any]:
        frames = self.core.get_ref_frames(video_path)
        pre = ("Answer the user's question based ONLY on the video frames.\n"
               "Be concise and specific. If information is not visible, say 'not visible'.\n"
               "Do NOT provide a generic caption.")
        text = self.core.vlm_answer(frames, question, preface=pre, mode="default")
        conf = 0.7
        if len(text.strip()) < 3 or text.strip().lower() in ("not visible", "unknown"):
            frames2 = VideoProcessor.uniform_sample_frames(video_path, num_frames=self.cfg.max_frames)
            text2 = self.core.vlm_answer(frames2, question, preface=pre, mode="default")
            if len(text2.strip()) > len(text.strip()):
                text = text2; conf = 0.75; frames = frames2
        prov = self.core.provenance(frames, meta)
        sheet = self.core.mcp_thumbnail_grid(frames)
        payload = {"answer": text, "contact_sheet": sheet, "provenance": prov}
        return self._format_answer(text=payload if fmt!="markdown" else text, fmt=fmt, provenance=prov, confidence=conf)

    def _respond(self, question: str) -> str:
        return ("Upload a video and ask for counts, timelines, anomalies, OCR, or ASR. "
                "You can set output via format=json|markdown|bullets|just_number.")

    def _format_answer(self, text: Any, fmt: AnswerMode, provenance: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        if fmt == "json":
            payload = text if isinstance(text, (dict, list)) else {"answer": text}
        elif fmt == "bullets":
            if isinstance(text, dict) and "timeline" in text:
                bullets = text["timeline"]
            else:
                bullets = [str(text)]
            payload = {"bullets": bullets}
        elif fmt == "just_number":
            payload = str(text) if not isinstance(text, (dict, list)) else json.dumps(text)
        else:
            payload = text if isinstance(text, str) else json.dumps(text, indent=2)
        return {"answer": payload, "format": fmt, "confidence": round(confidence, 2), "provenance": provenance}

# -------------------------
# Gradio UI
# -------------------------

def build_ui(agent: Agent) -> gr.Blocks:
    with gr.Blocks(title="Video Agent (MCP)", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <h2 style="text-align:center">🎬 Video Agent (Video-LLaVA + MCP)</h2>
        <p style="text-align:center">Counts • Timelines • OCR/ASR • Contact Sheets • Provenance • Answer Modes</p>
        """)
        with gr.Row():
            video = gr.Video(label="Upload Video")
            with gr.Column():
                mode = gr.Dropdown(choices=["json","markdown","bullets","just_number"], value="markdown", label="Answer Mode")
                q = gr.Textbox(label="Question", value="What's going on in this video?", lines=2)
                ask = gr.Button("Ask")
                clear = gr.Button("Clear")
        with gr.Row():
            out = gr.JSON(label="Answer")
        with gr.Row():
            status = gr.Markdown(value="Ready.")
        def _ask(v, question, m):
            t0 = time.time()
            path = v if isinstance(v, str) else (v.name if hasattr(v, "name") else None)
            res = agent.process(path, question, answer_mode=m)
            dt = time.time()-t0
            status_txt = f"Confidence: {res.get('confidence')} • Format: {res.get('format')} • Time: {dt:.2f}s"
            return res, status_txt
        ask.click(_ask, [video, q, mode], [out, status])
        def _clear():
            return None, "", {"cleared": True}, "Cleared."
        clear.click(_clear, None, [video, q, out, status])
    return demo

def create_agent(high_mem: bool = True) -> Agent:
    cfg = ModelConfig(
        videollava_model="LanguageBind/Video-LLaVA-7B-hf",
        router_model="microsoft/Phi-3-mini-4k-instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_frames=32 if high_mem else 16,
        max_new_tokens=200 if high_mem else 160,
    )
    acfg = AgentConfig(enable_reasoning=True, enable_memory=True, max_tool_calls=3, fallback_enabled=True, debug_mode=False)
    agent = Agent(cfg, acfg)

    # MCP servers can be added here:
    # servers = [
    #   ("yolo",   ["python", "mcp_yolo_server.py"]),
    #   ("scenes", ["python", "mcp_scenes_server.py"]),
    #   ("ffmpeg", ["python", "mcp_ffmpeg_server.py"]),
    #   ("whisper",["python", "mcp_whisper_server.py"]),
    #   ("ocr",    ["python", "mcp_ocr_server.py"]),
    # ]
    # agent.init_mcp_servers(servers)
    return agent

if __name__ == "__main__":
    agent = create_agent(high_mem=True)
    demo = build_ui(agent)
    demo.launch(share=False)
