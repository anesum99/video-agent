// Video Agent Chat Application

class VideoAgent {
    constructor() {
        this.socket = null;
        this.currentVideo = null;
        this.isProcessing = false;
        this.messageHistory = [];
        
        this.init();
    }

    init() {
        this.setupSocketIO();
        this.setupEventListeners();
        this.loadSettings();
        this.applyTheme();
    }

    setupSocketIO() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.showToast('Connected to server', 'success');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.showToast('Disconnected from server', 'error');
        });

        this.socket.on('processing_start', (data) => {
            this.showTypingIndicator();
        });

        this.socket.on('processing_chunk', (data) => {
            // Handle streaming responses if needed
            console.log('Chunk:', data);
        });

        this.socket.on('processing_complete', (data) => {
            this.hideTypingIndicator();
            this.addMessage(data.answer, 'bot');
        });

        this.socket.on('error', (data) => {
            this.hideTypingIndicator();
            this.showToast(data.message, 'error');
        });
    }

    setupEventListeners() {
        // Video upload
        const videoInput = document.getElementById('videoInput');
        const uploadArea = document.getElementById('uploadArea');
        
        videoInput.addEventListener('change', (e) => this.handleVideoUpload(e));
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                this.uploadVideo(files[0]);
            }
        });

        // Chat input
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        sendBtn.addEventListener('click', () => this.sendMessage());

        // Auto-resize textarea
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });

        // Quick action buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.dataset.question;
                document.getElementById('chatInput').value = question;
                this.sendMessage();
            });
        });

        // Clear button
        document.getElementById('clearBtn').addEventListener('click', () => this.clearSession());

        // Settings
        document.getElementById('settingsBtn').addEventListener('click', () => this.openSettings());

        // Theme selector
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.addEventListener('change', (e) => {
                localStorage.setItem('theme', e.target.value);
                this.applyTheme();
            });
        }
    }

    async handleVideoUpload(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('video/')) {
            await this.uploadVideo(file);
        }
    }

    async uploadVideo(file) {
        if (file.size > 100 * 1024 * 1024) {
            this.showToast('File too large. Maximum size is 100MB', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('video', file);

        this.showLoading('Uploading video...');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.currentVideo = data;
                this.showVideoPreview(file, data);
                this.showToast('Video uploaded successfully', 'success');
                this.addMessage(`Video "${data.filename}" uploaded successfully. You can now ask questions about it!`, 'bot');
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            this.showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    showVideoPreview(file, data) {
        const uploadArea = document.getElementById('uploadArea');
        const videoPreview = document.getElementById('videoPreview');
        const videoPlayer = document.getElementById('videoPlayer');
        const videoName = document.getElementById('videoName');
        const videoMetadata = document.getElementById('videoMetadata');

        uploadArea.classList.add('hidden');
        videoPreview.classList.remove('hidden');

        // Set video source
        videoPlayer.src = URL.createObjectURL(file);

        // Set video info
        videoName.textContent = data.filename;

        // Display metadata
        const metadata = data.metadata;
        videoMetadata.innerHTML = `
            <span><i class="fas fa-clock"></i> ${this.formatDuration(metadata.duration_seconds)}</span>
            <span><i class="fas fa-film"></i> ${metadata.fps} fps</span>
            <span><i class="fas fa-expand"></i> ${metadata.width}×${metadata.height}</span>
            <span><i class="fas fa-file-video"></i> ${metadata.codec}</span>
        `;
    }

    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message || this.isProcessing) return;

        // Add user message
        this.addMessage(message, 'user');
        
        // Clear input
        input.value = '';
        input.style.height = 'auto';

        // Get format
        const format = document.getElementById('formatSelect').value;

        // Send to server
        this.isProcessing = true;
        document.getElementById('sendBtn').disabled = true;
        this.showTypingIndicator();

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: message,
                    format: format
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.addMessage(data.answer, 'bot', data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            this.showToast(`Error: ${error.message}`, 'error');
            this.addMessage(`Sorry, I encountered an error: ${error.message}`, 'bot');
        } finally {
            this.isProcessing = false;
            document.getElementById('sendBtn').disabled = false;
            this.hideTypingIndicator();
        }
    }

    addMessage(content, sender, metadata = {}) {
        const messagesContainer = document.getElementById('chatMessages');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = sender === 'bot' ? 
            '<i class="fas fa-robot"></i>' : 
            '<i class="fas fa-user"></i>';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Format content based on type
        if (typeof content === 'object') {
            contentDiv.innerHTML = `<pre>${JSON.stringify(content, null, 2)}</pre>`;
        } else {
            // Parse markdown-like formatting
            contentDiv.innerHTML = this.formatMessage(content);
        }
        
        // Add confidence badge if available
        if (metadata.confidence) {
            const confidenceBadge = document.createElement('div');
            confidenceBadge.className = 'confidence-badge';
            confidenceBadge.style.cssText = `
                margin-top: 0.5rem;
                font-size: 0.75rem;
                color: var(--text-tertiary);
            `;
            confidenceBadge.textContent = `Confidence: ${(metadata.confidence * 100).toFixed(0)}%`;
            contentDiv.appendChild(confidenceBadge);
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Save to history
        this.messageHistory.push({
            sender,
            content,
            metadata,
            timestamp: new Date().toISOString()
        });
    }

    formatMessage(text) {
        // Simple markdown parsing
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/• /g, '• ');
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatMessages');
        
        if (document.getElementById('typingIndicator')) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typingIndicator';
        typingDiv.className = 'message bot-message';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    async clearSession() {
        if (!confirm('Clear current session and start over?')) return;

        try {
            const response = await fetch('/clear', {
                method: 'POST'
            });

            if (response.ok) {
                // Clear UI
                document.getElementById('chatMessages').innerHTML = `
                    <div class="message bot-message">
                        <div class="message-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="message-content">
                            <p>👋 Session cleared! Upload a new video to start.</p>
                        </div>
                    </div>
                `;
                
                // Reset video preview
                document.getElementById('uploadArea').classList.remove('hidden');
                document.getElementById('videoPreview').classList.add('hidden');
                document.getElementById('videoInput').value = '';
                
                this.currentVideo = null;
                this.messageHistory = [];
                
                this.showToast('Session cleared', 'success');
            }
        } catch (error) {
            this.showToast('Failed to clear session', 'error');
        }
    }

    openSettings() {
        document.getElementById('settingsModal').classList.remove('hidden');
    }

    closeSettings() {
        document.getElementById('settingsModal').classList.add('hidden');
    }

    loadSettings() {
        const theme = localStorage.getItem('theme') || 'auto';
        const maxFrames = localStorage.getItem('maxFrames') || '16';
        const model = localStorage.getItem('model') || 'gemini-1.5-flash';

        document.getElementById('themeSelect').value = theme;
        document.getElementById('maxFrames').value = maxFrames;
        document.getElementById('modelSelect').value = model;
    }

    applyTheme() {
        const theme = localStorage.getItem('theme') || 'auto';
        
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        } else {
            document.documentElement.setAttribute('data-theme', theme);
        }
    }

    showLoading(text = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        
        loadingText.textContent = text;
        overlay.classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        }[type] || 'fa-info-circle';
        
        toast.innerHTML = `
            <i class="fas ${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease forwards';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }
}

// Global functions for modal
window.closeSettings = () => {
    document.getElementById('settingsModal').classList.add('hidden');
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.videoAgent = new VideoAgent();
});