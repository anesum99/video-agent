# Video Analysis Agent 🎥

A powerful multimodal video analysis system that combines Google Gemini AI with specialized microservices to provide comprehensive video understanding capabilities. The system uses a modular architecture with HTTP-based microservices for scalable video processing.

## 🌟 Features

### Core Capabilities
- **Visual Analysis**: Describe what's happening in videos using Google Gemini
- **Object Detection & Counting**: Count people, vehicles, and 80+ object types using YOLOv10
- **Text Extraction (OCR)**: Extract and read text from video frames using EasyOCR
- **Speech Transcription (ASR)**: Transcribe audio and speech using OpenAI Whisper
- **Scene Detection**: Identify and analyze distinct scenes and shot changes
- **Timeline Generation**: Create step-by-step breakdowns of video events
- **Metadata Extraction**: Get video duration, resolution, FPS, and codec information
- **Interactive Chat Interface**: Web-based UI with real-time video analysis

### Technical Features
- **Microservices Architecture**: Modular design with independent services
- **Intelligent Routing**: Phi3-powered semantic routing for optimal task selection
- **Multiple Output Formats**: JSON, Markdown, Bullets, or plain numbers
- **Memory Optimization**: Automatic model loading/unloading for efficient resource usage
- **Session Management**: Redis-backed session storage with automatic cleanup
- **WebSocket Support**: Real-time streaming responses
- **Docker Deployment**: Containerized services for easy deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   Web Interface                  │
│              (Flask + SocketIO + UI)             │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│                 Main Application                 │
│              (app.py + Gemini Core)              │
└─────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    Router    │ │ HTTP Manager │ │   Redis      │
│  (Phi3/LLM)  │ │   Manager    │ │  Sessions    │
└──────────────┘ └──────────────┘ └──────────────┘
                         │
    ┌────────┬──────────┼──────────┬────────┐
    ▼        ▼          ▼          ▼        ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│  YOLO  ││Whisper ││  OCR   ││ Scenes ││ FFmpeg │
│(Object)││ (ASR)  ││ (Text) ││(Detect)││ (Video)│
└────────┘└────────┘└────────┘└────────┘└────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Google Gemini API key
- 4GB+ RAM available
- Python 3.10+ (for local development)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/video-agent.git
cd video-agent
```

2. **Set up environment variables**:
```bash
# Create .env file
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_secret_key_here
EOF
```

3. **Deploy with Docker Compose**:

**Development Deployment (Recommended)**:
```bash
# Deploy the development environment
docker-compose -f docker-compose-dev.yml up -d
```

**Production Deployment Options**:
```bash
# Option A: Full Microservices
docker-compose -f docker-compose-microservices.yml up -d

# Option B: Lightweight Bundle
docker-compose -f docker-compose-lightweight.yml up -d
```

4. **Access the application**:
- **Development Environment**: `http://localhost:8080`
- **Production**: `http://localhost:80`

## 📦 Services & Components

### Main Application (`app.py`)
- Flask web server with SocketIO support
- Handles video uploads and user interactions
- Coordinates with Gemini API and microservices
- Session management with Redis

### Video Agent Core (`video_agent_gemini.py`)
- LangGraph-based workflow orchestration
- Google Gemini integration for visual analysis
- Frame extraction and caching
- Answer formatting and confidence scoring

### Microservices

| Service | Dev Port | Prod Port | Purpose | Model/Tech |
|---------|----------|-----------|---------|------------|
| **Router** | 8016 | 8006 | Intent classification | Phi3:mini via Ollama |
| **YOLO** | 8011 | 8001 | Object detection | YOLOv10m ONNX |
| **Whisper** | 8012 | 8002 | Speech transcription | OpenAI Whisper |
| **OCR** | 8013 | 8003 | Text extraction | EasyOCR |
| **Scenes** | 8014 | 8004 | Scene detection | Custom algorithm |
| **FFmpeg** | 8015 | 8005 | Video processing | FFmpeg |
| **Web App** | 5001 | 5000 | Main Flask application | Flask + SocketIO |
| **Nginx** | 8080 | 80 | Reverse proxy | Nginx |
| **Redis** | 6380 | 6379 | Session storage | Redis |
| **Ollama** | 11435 | 11434 | LLM service | Ollama |

### HTTP Tool Manager (`http_tool_manager.py`)
- Service discovery and health checking
- Request routing and load balancing
- Fallback handling for service failures

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `SECRET_KEY` | Flask session secret | Auto-generated |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `MAX_FRAMES` | Max frames to extract | `16` |
| `GEMINI_MODEL` | Gemini model version | `gemini-1.5-flash` |
| `MAX_CONCURRENT_SESSIONS` | Session limit | `5` |

### Service URLs (for custom deployments)

| Variable | Description | Default |
|----------|-------------|---------|
| `ROUTER_URL` | Router service URL | `http://router:8006` |
| `YOLO_URL` | YOLO service URL | `http://yolo:8001` |
| `WHISPER_URL` | Whisper service URL | `http://whisper:8002` |
| `OCR_URL` | OCR service URL | `http://ocr:8003` |
| `SCENES_URL` | Scenes service URL | `http://scenes:8004` |
| `FFMPEG_URL` | FFmpeg service URL | `http://ffmpeg:8005` |

## 📝 API Reference

### Web Application Endpoints

#### `GET /`
Returns the main web interface

#### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "redis": true,
  "agent": true
}
```

#### `POST /upload`
Upload a video file
- **Body**: `multipart/form-data` with video file
- **Response**: Video metadata and thumbnail

#### `POST /analyze`
Analyze video with a question
```json
{
  "question": "How many people are in the video?",
  "format": "markdown|json|bullets|just_number"
}
```

#### `POST /clear`
Clear current session and uploaded video

### Microservice Tool Endpoints

Each microservice exposes tools via HTTP:

#### YOLO Service
- `POST /tools/count_objects` - Count specific objects
- `POST /tools/detect_objects` - Detect all objects

#### Whisper Service
- `POST /tools/transcribe_video` - Transcribe audio

#### OCR Service
- `POST /tools/extract_text` - Extract text from frames

#### Scenes Service
- `POST /tools/detect_scenes` - Identify scene changes

#### Router Service
- `POST /classify` - Classify user intent

## 🎯 Usage Examples

### Basic Usage
1. Open the web interface
2. Upload a video (MP4, AVI, MOV, WebM, MKV)
3. Ask questions like:
   - "What's happening in this video?"
   - "How many people are there?"
   - "Break this down step by step"
   - "What text is visible?"
   - "What are they saying?"

### Advanced Features
- **Format Control**: Select output format (Text, Bullets, JSON, Number)
- **Quick Actions**: Use preset questions for common tasks
- **Real-time Updates**: See processing status via WebSocket

### Programmatic Usage
```python
from video_agent_gemini import create_agent, AnswerFormat

# Create agent
agent = create_agent(api_key="your_gemini_api_key")

# Process video
result = agent.process(
    video_path="path/to/video.mp4",
    question="How many cars are in the video?",
    answer_format=AnswerFormat.JUST_NUMBER
)

print(result)  # Output: "5"
```

## 🐳 Docker Deployment

### Build Images
```bash
# Build all services
./build.sh

# Or build individually
docker build -f Dockerfile.yolo -t video-agent-yolo .
docker build -f Dockerfile.whisper -t video-agent-whisper .
# ... etc
```

### Deploy to Production
```bash
# Deploy with resource limits
docker-compose -f docker-compose-microservices.yml up -d

# Monitor services
docker-compose logs -f

# Scale specific services
docker-compose up -d --scale yolo=2
```

### Resource Requirements

| Deployment | RAM | CPU | Storage |
|------------|-----|-----|---------|
| Full Microservices | 8GB+ | 4+ cores | 10GB |
| Lightweight Bundle | 4GB+ | 2+ cores | 5GB |
| Development | 2GB+ | 2+ cores | 3GB |

## 🛠️ Development

### Local Setup
```bash
# Install Python dependencies
pip install -r requirements_main.txt

# Run main application
python app.py

# Run individual services
python yolo_http_server.py
python whisper_http_server.py
# ... etc
```

### Testing Services
```bash
# Development deployment
./deploy-dev.sh

# Run individual service tests
python yolo_http_server_dev.py
python ocr_http_server_dev.py
# ... etc
```

### Adding New Services
1. Create service file: `service_http_server.py`
2. Inherit from `HTTPServer` base class
3. Add tool endpoints
4. Create Dockerfile
5. Update docker-compose configuration

## 🔍 Troubleshooting

### Common Issues

**1. Service not available**
- Check service health: `curl http://localhost:PORT/health`
- Check logs: `docker-compose logs service-name`
- Restart service: `docker-compose restart service-name`

**2. Out of memory**
- Reduce `MAX_FRAMES` environment variable
- Enable memory optimization in services
- Scale down concurrent services

**3. Slow processing**
- Check network connectivity between services
- Verify model loading times
- Consider using lighter models (gemini-1.5-flash)

**4. Upload fails**
- Check file size (max 100MB by default)
- Verify supported format (MP4, AVI, MOV, WebM, MKV)
- Check disk space for uploads folder

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Google Gemini for multimodal AI capabilities
- YOLOv10 for object detection
- OpenAI Whisper for speech recognition
- EasyOCR for text extraction
- LangGraph for workflow orchestration
- Flask & SocketIO for web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Built with ❤️ for the AI community