# Video Agent with MCP Integration

A powerful video analysis agent that combines Video-LLaVA with MCP (Model Context Protocol) tools for comprehensive video understanding. Features deterministic object counting, scene detection, OCR, ASR, and structured output formats.

## 🎬 Features

- **Multi-modal Analysis**: Visual Q&A, object counting, timeline generation
- **MCP Tool Integration**: YOLO, Whisper, OCR, scene detection, FFmpeg
- **Flexible Output Formats**: JSON, Markdown, Bullets, Just Numbers
- **Smart Routing**: Automatically selects the best tool for each query
- **Confidence & Provenance**: Tracks which frames were used and confidence levels
- **Interactive UI**: Gradio interface with format selection

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-agent-mcp.git
cd video-agent-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Agent

```bash
# Launch the Gradio UI
python video_agent_mcp_full.py

# With options
python video_agent_mcp_full.py --low-memory  # For systems with <16GB RAM
python video_agent_mcp_full.py --share       # Share publicly via Gradio
```

## 📝 Example Queries

```python
# Basic questions
"What's happening in this video?"
"How many people are in the video?"
"Break this down step by step"

# With format control
"How many bikes? format=just_number"  # Returns: 2
"What's the timeline? format=bullets" # Returns: bullet list
"Describe the scene format=json"      # Returns: structured JSON

# Advanced queries
"Count only foreground people"
"What text is visible in the frames?"
"What are they saying?" (requires Whisper MCP)
```

## 🔧 MCP Server Setup (Optional)

MCP servers provide deterministic tools for better accuracy. The agent works without them but performs better with them enabled.

### Enable MCP Servers

Edit `video_agent_mcp_full.py` and uncomment the servers list:

```python
servers = [
    ("yolo", ["python", "mcp_yolo_server.py"]),
    ("whisper", ["python", "mcp_whisper_server.py"]),
    ("ocr", ["python", "mcp_ocr_server.py"]),
    # ... more servers
]
```

### Install Additional Dependencies

```bash
# For YOLO object detection
pip install ultralytics

# For audio transcription
pip install openai-whisper

# For OCR
pip install easyocr
# or
pip install pytesseract

# For scene detection
pip install scenedetect[opencv]
```

## 📊 Answer Formats

| Format | Description | Example Output |
|--------|-------------|----------------|
| `json` | Structured data with metadata | `{"answer": "...", "confidence": 0.85, "provenance": {...}}` |
| `markdown` | Human-readable text | "There are **2 people** exercising..." |
| `bullets` | Timeline as bullet list | "• Person enters frame\n• Starts running..." |
| `just_number` | Numeric only for counts | `2` or `2-3` |

## 🏗️ Architecture

```
User Query → Intent Router → Frame Extraction → Analysis
                ↓                                   ↓
          Determine Tool                    MCP Tools (if available)
                ↓                                   ↓
          Select Format                      VLM Fallback
                ↓                                   ↓
           Verify Pass → Format Output → Return Result
```

## 📁 Project Structure

```
video-agent-mcp/
├── video_agent_mcp_full.py      # Main agent implementation
├── mcp_yolo_server.py           # YOLO object detection server
├── mcp_whisper_server.py        # Whisper ASR server
├── mcp_ocr_server.py            # OCR text extraction server
├── mcp_scenes_server.py         # Scene detection server
├── mcp_ffmpeg_server.py         # Video processing utilities
├── videoAgentLatest.ipynb       # Original notebook
├── video_agent_mcp_clone.py     # Initial clone version
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── README_VIDEO_AGENT.md        # Detailed documentation
```

## 🎯 Use Cases

- **Content Analysis**: Understand what's happening in videos
- **Object Counting**: Count people, vehicles, objects
- **Timeline Creation**: Generate step-by-step breakdowns
- **Text Extraction**: Read signs, labels, subtitles
- **Speech Transcription**: Convert speech to text
- **Scene Detection**: Identify cuts and transitions

## ⚡ Performance Tips

- Use `--low-memory` flag for systems with limited RAM
- Enable GPU with CUDA for 5-10x speedup
- Install OpenCV for better keyframe extraction
- Use MCP servers for deterministic operations

## 📖 Documentation

See [README_VIDEO_AGENT.md](README_VIDEO_AGENT.md) for detailed documentation including:
- Complete API reference
- Configuration options
- Troubleshooting guide
- Extension instructions

## 🤝 Contributing

Contributions are welcome! To extend the agent:

1. Add new tool types in the `ToolType` enum
2. Create analysis nodes in `VideoAgent`
3. Update router logic for intent detection
4. Optionally add MCP servers for deterministic processing

## 📄 License

This project uses open-source models and libraries. See individual component licenses.

## 🙏 Acknowledgments

- Video-LLaVA model by LanguageBind
- MCP protocol by Anthropic
- Various open-source libraries (see requirements.txt)