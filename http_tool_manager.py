#!/usr/bin/env python3
"""
HTTP Tool Manager - Simple lazy loading approach
Only tests connectivity when actually needed, not during import
"""

import os
import requests
import logging
from typing import Dict, Any, Optional
import time

class HTTPToolManager:
    """Manages microservices via HTTP with lazy connectivity testing"""
    
    def __init__(self):
        self.services = {
            'router': os.getenv('ROUTER_URL', 'http://router:8006'),
            'yolo': os.getenv('YOLO_URL', 'http://yolo:8001'),
            'whisper': os.getenv('WHISPER_URL', 'http://whisper:8002'),
            'ocr': os.getenv('OCR_URL', 'http://ocr:8003'),
            'scenes': os.getenv('SCENES_URL', 'http://scenes:8004'),
            'ffmpeg': os.getenv('FFMPEG_URL', 'http://ffmpeg:8005')
        }
        self.enabled = True
        self.timeout = 10
        self._connectivity_tested = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("http-tool-manager")
        
        # DON'T test connectivity here - do it lazily when first needed
        self.logger.info("HTTP Tool Manager initialized - will test connectivity on first use")
    
    def _test_connectivity_once(self):
        """Test connectivity only once, when first needed"""
        if self._connectivity_tested:
            return
        
        self.logger.info("Testing microservice connectivity (first use)...")
        
        for service_name, url in self.services.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"✅ {service_name} service available")
                else:
                    self.logger.warning(f"⚠️ {service_name} service returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"⚠️ {service_name} service not available: {str(e)}")
        
        self._connectivity_tested = True
    
    def is_service_available(self, service_name: str) -> bool:
        """Check if a service is available"""
        if service_name not in self.services:
            return False
        
        try:
            url = self.services[service_name]
            response = requests.get(f"{url}/health", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def call_tool(self, server: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool via HTTP"""
        if not self.enabled:
            return {"error": "HTTP services disabled"}
        
        if server not in self.services:
            return {"error": f"Unknown service: {server}"}
        
        # Test connectivity on first tool call
        self._test_connectivity_once()
        
        # Quick availability check
        if not self.is_service_available(server):
            return {"error": f"Service {server} not available"}
        
        try:
            url = f"{self.services[server]}/tools/{tool}"
            
            self.logger.debug(f"Calling {server}.{tool}")
            
            response = requests.post(
                url, 
                json=args,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.debug(f"Tool call successful: {server}.{tool}")
                return result
            else:
                error_msg = f"Tool call failed: {server}.{tool} returned {response.status_code}"
                self.logger.error(error_msg)
                return {"error": error_msg}
        
        except requests.exceptions.Timeout:
            error_msg = f"Tool call timeout: {server}.{tool}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        except Exception as e:
            error_msg = f"Tool call error: {server}.{tool} - {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def call_tool_sync(self, server: str, tool: str, args: dict) -> dict:
        """Call a tool synchronously with logging"""
        
        if not self._connectivity_tested:
            self._test_connectivity_once()
        
        if server not in self.services:
            self.logger.warning(f"🚫 SERVICE UNAVAILABLE: {server}")
            return {"error": f"Service {server} not available"}
        
        url = f"{self.services[server]}/tools/{tool}"
        self.logger.info(f"📞 CALLING: {server}.{tool} at {url}")
        
        try:
            response = requests.post(url, json=args, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"✅ SERVICE SUCCESS: {server}.{tool} completed")
                return result
            else:
                self.logger.error(f"❌ SERVICE ERROR: {server}.{tool} returned {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            self.logger.error(f"❌ SERVICE EXCEPTION: {server}.{tool} failed - {e}")
            return {"error": str(e)}
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of a specific service"""
        if service_name not in self.services:
            return {"error": f"Unknown service: {service_name}"}
        
        # Test connectivity if not done yet
        self._test_connectivity_once()
        
        try:
            url = f"{self.services[service_name]}/tools/status"
            response = requests.post(url, json={}, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Service status call failed: {response.status_code}"}
        
        except Exception as e:
            return {"error": f"Service status error: {str(e)}"}
    
    def get_all_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}
        for service_name in self.services.keys():
            status[service_name] = self.get_service_status(service_name)
        return status

