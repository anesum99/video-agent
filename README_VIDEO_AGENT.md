# Video Agent with MCP Integration

A powerful video analysis system combining Video-LLaVA with MCP (Model Context Protocol) tools for deterministic counting, scene detection, OCR, ASR, and more.

## Features

### Core Capabilities
- **Visual Q&A**: Answer questions about video content using Video-LLaVA
- **Object Counting**: Deterministic counting with YOLO (via MCP) or VLM fallback
- **Timeline Generation**: Step-by-step breakdown of video events
- **Metadata Extraction**: Duration, resolution, FPS, codec information
- **Scene Detection**: Identify shot boundaries and scene changes
- **OCR**: Extract text from video frames
- **ASR**: Transcribe speech from video audio
- **Contact Sheets**: Auto-generated thumbnail grids

### Answer Formats
- `json`: Structured data with provenance
- `markdown`: Human-readable text
- `bullets`: Timeline as bullet list
- `just_number`: Strict numeric output for counts

### UI/UX Features
- **Confidence Scoring**: Each answer includes confidence level
- **Provenance Tracking**: Shows which frames/timestamps were used
- **Follow-up Questions**: Handles ambiguity (e.g., "count foreground or all people?")
- **Frame Caching**: Consistent results across multiple queries
- **Verify Pass**: Automatic retry with uniform frames for low-confidence answers

## Installation

### Basic Setup
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers av gradio langgraph langchain-core pillow numpy

# Optional: Better keyframe extraction
pip install opencv-python

# Optional: MCP support
pip install mcp
```

### GPU Support
For CUDA support, ensure you have NVIDIA drivers and CUDA toolkit installed, then:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Run the Gradio UI
```bash
python video_agent_mcp_full.py
```
This launches a web interface at http://localhost:7860

### 2. Command Line Options
```bash
# Use low memory mode (16 frames max)
python video_agent_mcp_full.py --low-memory

# Share publicly via Gradio
python video_agent_mcp_full.py --share

# Custom port
python video_agent_mcp_full.py --port 8080
```

### 3. Programmatic Usage
```python
from video_agent_mcp_full import create_agent, AnswerFormat

# Create agent
agent = create_agent(high_memory=True)

# Process video
result = agent.process(
    video_path="path/to/video.mp4",
    question="How many people are in the video?",
    answer_format=AnswerFormat.JUST_NUMBER
)

print(result)  # Output: "2"
```

## Example Queries

### Basic Questions
- "What's happening in this video?"
- "How many people are in the video?"
- "Break this down step by step"
- "What text is visible?"
- "What are they saying?"

### With Format Control
- "How many bikes? format=just_number" → `1`
- "What's the timeline? format=bullets" → Bullet list
- "Describe the scene format=json" → Structured JSON

### Advanced Queries
- "Count only foreground people"
- "What unusual things happen?"
- "Read the sign in the background"
- "List all scene changes"

## MCP Server Setup (Optional)

MCP servers provide deterministic tools for better accuracy. To enable:

### 1. Install Server Dependencies
```bash
# For YOLO counting
pip install ultralytics

# For OCR
pip install easyocr
# or
pip install pytesseract

# For ASR
pip install openai-whisper

# For scene detection
pip install scenedetect[opencv]

# For video processing
pip install ffmpeg-python
```

### 2. Enable Servers in Code
Edit `video_agent_mcp_full.py` and uncomment the servers:

```python
servers = [
    ("yolo", ["python", "mcp_yolo_server.py"]),
    ("scenes", ["python", "mcp_scenes_server.py"]),
    ("ffmpeg", ["python", "mcp_ffmpeg_server.py"]),
    ("whisper", ["python", "mcp_whisper_server.py"]),
    ("ocr", ["python", "mcp_ocr_server.py"]),
]
```

### 3. Implement Server Logic
The provided MCP server files are stubs. Replace the TODO sections with actual implementations:

```python
# In mcp_yolo_server.py
model = YOLO('yolov8n.pt')
results = model(img)
# ... actual counting logic
```

## Architecture

### Components
1. **VideoProcessor**: Handles frame extraction and metadata
2. **IntentRouter**: Classifies questions and determines tool usage
3. **VideoLLaVACore**: Manages the Video-LLaVA model
4. **MCPToolManager**: Interfaces with MCP servers
5. **VideoAgent**: Orchestrates the workflow using LangGraph

### Workflow
```
Question → Router → Extract Frames → Analysis Node → Verify → Format Output
                ↓
           MCP Tools (if available)
                ↓
           VLM Fallback
```

## Performance Tips

### Memory Management
- Use `--low-memory` flag for systems with <16GB RAM
- Reduce `max_frames` in ModelConfig for faster processing
- Clear frame cache periodically in long-running sessions

### Speed Optimization
- Enable GPU with CUDA for 5-10x speedup
- Use MCP servers for deterministic operations
- Cache Video-LLaVA model between requests

### Accuracy Improvements
- Install OpenCV for better keyframe extraction
- Use higher `max_frames` for complex videos
- Enable MCP YOLO for accurate counting
- Adjust confidence thresholds in AgentConfig

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Use `--low-memory` flag
   - Reduce batch size or max_frames
   - Use CPU mode if GPU memory insufficient

2. **Slow Processing**
   - Ensure CUDA is properly installed for GPU
   - Check that models are cached (not reloading)
   - Reduce max_frames or video resolution

3. **MCP Connection Failed**
   - Verify MCP package is installed
   - Check server scripts are executable
   - Ensure Python path is correct in server commands

4. **Poor Counting Accuracy**
   - Enable YOLO MCP server for deterministic counts
   - Increase max_frames for better coverage
   - Use "include background" or "foreground only" in queries

## API Reference

### Main Functions

```python
create_agent(high_memory: bool = True) -> VideoAgent
```
Creates and configures the video agent.

```python
agent.process(
    video_path: Optional[str],
    question: str,
    answer_format: AnswerFormat = AnswerFormat.MARKDOWN
) -> Dict[str, Any]
```
Process a video question and return results.

### Answer Formats
- `AnswerFormat.JSON`: Structured data
- `AnswerFormat.MARKDOWN`: Human-readable text
- `AnswerFormat.BULLETS`: Bullet list
- `AnswerFormat.JUST_NUMBER`: Numeric only

### Configuration Classes
- `ModelConfig`: Model paths, device, frame limits
- `AgentConfig`: Reasoning, memory, confidence settings

## License

This project uses:
- Video-LLaVA model (Apache 2.0)
- Various open-source libraries (see requirements)

## Contributing

To extend the agent:
1. Add new tool types in `ToolType` enum
2. Create analysis node in `VideoAgent`
3. Update router logic for intent detection
4. Optionally add MCP server for deterministic processing