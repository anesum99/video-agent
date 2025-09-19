#!/usr/bin/env python3
"""
Base HTTP server for microservices
Pure Flask-based REST API server for microservices
"""

import json
import logging
from typing import Dict, Any, Optional, Callable
from flask import Flask, request, jsonify

class HTTPServer:
    """Base class for HTTP microservice servers"""
    
    def __init__(self, service_name: str, port: int = 8000):
        self.service_name = service_name
        self.port = port
        self.app = Flask(f"{service_name}-service")
        self.tools = {}  # Track available tools
        self.setup_routes()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"{service_name}-http")
    
    def setup_routes(self):
        """Setup standard HTTP routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'service': self.service_name,
                'status': 'healthy'
            })
        
        @self.app.route('/tools', methods=['GET'])
        def list_tools():
            """List available tools"""
            return jsonify({
                'service': self.service_name,
                'tools': list(self.tools.keys())
            })
        
        @self.app.route('/tools/status', methods=['POST'])
        def tools_status():
            """Status endpoint for HTTP tool manager compatibility"""
            return jsonify({
                'service': self.service_name,
                'status': 'healthy',
                'tools': list(self.tools.keys())
            })
    
    def add_tool_endpoint(self, tool_name: str, handler: Callable):
        """Add a tool endpoint"""
        endpoint = f"/tools/{tool_name}"
        
        def tool_handler():
            try:
                data = request.get_json() or {}
                
                # Call the handler function with the request data
                result = handler(**data)
                
                return jsonify(result)
                
            except Exception as e:
                self.logger.error(f"Error in {tool_name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Add the route
        self.app.add_url_rule(
            endpoint, 
            f"tool_{tool_name}", 
            tool_handler, 
            methods=['POST']
        )
        
        # Track the tool
        self.tools[tool_name] = handler
        
        self.logger.info(f"Added endpoint: {endpoint}")
    
    def run(self, host: str = '0.0.0.0', debug: bool = False):
        """Run the HTTP server"""
        self.logger.info(f"Starting {self.service_name} HTTP server on {host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug, threaded=True)

