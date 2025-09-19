#!/usr/bin/env python3
"""
Flask web application for Gemini Video Agent
Lightweight and optimized for AWS EC2 free tier
Updated to use HTTP microservices
"""

import os
import json
import time
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import redis

from video_agent_gemini import create_agent, AnswerFormat, VideoProcessor

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()
    UPLOAD_FOLDER = Path('uploads')
    PROCESSED_FOLDER = Path('processed')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    SESSION_LIFETIME = 3600  # 1 hour
    
    # Gemini configuration
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
    
    # Performance settings for free tier
    MAX_FRAMES = int(os.environ.get('MAX_FRAMES', '16'))  # Reduced for free tier
    MAX_CONCURRENT_SESSIONS = int(os.environ.get('MAX_CONCURRENT_SESSIONS', '5'))

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
# Revert back to eventlet for WebSocket support - no more asyncio conflicts
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Create directories
Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
Config.PROCESSED_FOLDER.mkdir(exist_ok=True)

# Initialize Redis for session management (optional, falls back to in-memory)
try:
    redis_client = redis.from_url(Config.REDIS_URL)
    redis_client.ping()
    print("✅ Redis connected")
except:
    redis_client = None
    print("⚠️  Redis not available, using in-memory sessions")

# In-memory session store (fallback)
sessions_store = {}

# Initialize video agent
agent = None

def get_agent():
    """Lazy load the agent"""
    global agent
    if agent is None:
        agent = create_agent(api_key=Config.GEMINI_API_KEY)
    return agent

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_session_data(session_id: str) -> Dict[str, Any]:
    """Get session data from Redis or memory"""
    if redis_client:
        try:
            data = redis_client.get(f"session:{session_id}")
            return json.loads(data) if data else {}
        except:
            pass
    return sessions_store.get(session_id, {})

def set_session_data(session_id: str, data: Dict[str, Any]):
    """Set session data in Redis or memory"""
    if redis_client:
        try:
            redis_client.setex(
                f"session:{session_id}",
                Config.SESSION_LIFETIME,
                json.dumps(data)
            )
            return
        except:
            pass
    sessions_store[session_id] = data

def cleanup_old_sessions():
    """Clean up old sessions from memory"""
    if not redis_client and len(sessions_store) > Config.MAX_CONCURRENT_SESSIONS:
        # Remove oldest sessions
        sorted_sessions = sorted(
            sessions_store.items(),
            key=lambda x: x[1].get('last_activity', 0)
        )
        for session_id, _ in sorted_sessions[:-Config.MAX_CONCURRENT_SESSIONS]:
            del sessions_store[session_id]
            # Also cleanup associated files
            cleanup_session_files(session_id)

def cleanup_session_files(session_id: str):
    """Clean up uploaded files for a session"""
    session_data = get_session_data(session_id)
    if 'video_path' in session_data:
        try:
            Path(session_data['video_path']).unlink(missing_ok=True)
        except:
            pass
    if 'contact_sheet' in session_data:
        try:
            Path(session_data['contact_sheet']).unlink(missing_ok=True)
        except:
            pass

# Routes

@app.route('/')
def index():
    """Serve the main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.min.html')

@app.route('/health')
def health():
    """Health check endpoint for AWS"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'redis': redis_client is not None,
        'agent': agent is not None
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = f"{session_id}_{int(time.time())}_{filename}"
    filepath = Config.UPLOAD_FOLDER / unique_filename
    file.save(str(filepath))
    
    # Extract metadata and create thumbnail
    try:
        metadata = VideoProcessor.get_video_metadata(str(filepath))
        
        # Extract a single frame for thumbnail
        frames = VideoProcessor.uniform_sample_frames(str(filepath), num_frames=1)
        if frames:
            thumbnail_path = Config.PROCESSED_FOLDER / f"{session_id}_thumb.jpg"
            frames[0].save(str(thumbnail_path), 'JPEG', quality=85)
            metadata['thumbnail'] = f"/thumbnail/{session_id}"
        
        # Update session
        session_data = get_session_data(session_id)
        session_data.update({
            'video_path': str(filepath),
            'video_name': filename,
            'metadata': metadata,
            'last_activity': time.time()
        })
        set_session_data(session_id, session_data)
        
        # Cleanup old sessions
        cleanup_old_sessions()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'metadata': metadata
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/thumbnail/<session_id>')
def get_thumbnail(session_id):
    """Serve video thumbnail"""
    thumbnail_path = Config.PROCESSED_FOLDER / f"{session_id}_thumb.jpg"
    if thumbnail_path.exists():
        return send_file(str(thumbnail_path), mimetype='image/jpeg')
    return '', 404

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze video with a question"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    data = request.json
    question = data.get('question', '')
    format_type = data.get('format', 'markdown')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    session_data = get_session_data(session_id)
    video_path = session_data.get('video_path')
    
    if not video_path or not Path(video_path).exists():
        return jsonify({
            'answer': 'Please upload a video first. I can help you analyze videos by answering questions about their content.',
            'confidence': 1.0,
            'no_video': True
        })
    
    try:
        # Convert format
        format_map = {
            'markdown': AnswerFormat.MARKDOWN,
            'json': AnswerFormat.JSON,
            'bullets': AnswerFormat.BULLETS,
            'just_number': AnswerFormat.JUST_NUMBER
        }
        answer_format = format_map.get(format_type, AnswerFormat.MARKDOWN)
        
        # Process with agent
        start_time = time.time()
        agent = get_agent()
        result = agent.process(video_path, question, answer_format)
        elapsed = time.time() - start_time
        
        # Format response
        if isinstance(result, dict):
            response = {
                'answer': result.get('answer') or result.get('message') or json.dumps(result),
                'confidence': result.get('confidence', 0.5),
                'type': result.get('type', 'analysis'),
                'processing_time': round(elapsed, 2)
            }
        else:
            response = {
                'answer': str(result),
                'confidence': 0.7,
                'processing_time': round(elapsed, 2)
            }
        
        # Update session activity
        session_data['last_activity'] = time.time()
        set_session_data(session_id, session_data)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'answer': f'Sorry, I encountered an error: {str(e)}'
        }), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clear the current session"""
    session_id = session.get('session_id')
    if session_id:
        cleanup_session_files(session_id)
        if redis_client:
            redis_client.delete(f"session:{session_id}")
        elif session_id in sessions_store:
            del sessions_store[session_id]
    
    # Generate new session
    session['session_id'] = str(uuid.uuid4())
    return jsonify({'success': True, 'new_session': session['session_id']})

# WebSocket events for real-time chat

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to video agent'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('analyze_stream')
def handle_analyze_stream(data):
    """Stream analysis results"""
    session_id = data.get('session_id')
    question = data.get('question')
    
    if not session_id or not question:
        emit('error', {'message': 'Missing session or question'})
        return
    
    # Start processing indicator
    emit('processing_start', {'question': question})
    
    try:
        session_data = get_session_data(session_id)
        video_path = session_data.get('video_path')
        
        if not video_path:
            emit('processing_complete', {
                'answer': 'Please upload a video first.',
                'no_video': True
            })
            return
        
        # Process
        agent = get_agent()
        result = agent.process(video_path, question, AnswerFormat.MARKDOWN)
        
        # Stream the response
        if isinstance(result, str):
            # Simulate streaming by sending chunks
            chunks = result.split('. ')
            for i, chunk in enumerate(chunks):
                emit('processing_chunk', {
                    'chunk': chunk + ('.' if i < len(chunks) - 1 else ''),
                    'progress': (i + 1) / len(chunks)
                })
                socketio.sleep(0.1)  # Small delay for effect
        
        emit('processing_complete', {'answer': result})
        
    except Exception as e:
        emit('error', {'message': str(e)})

# Error handlers

@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

# No ASGI wrapper needed - using eventlet workers

if __name__ == '__main__':
    # Development server
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
