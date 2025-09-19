import tempfile
import atexit
from pathlib import Path
from typing import Set
import logging

class TempFileManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.temp_files = set()
            atexit.register(cls._instance.cleanup_all)
        return cls._instance
    
    def create_temp_file(self, suffix='', prefix='tmp', dir=None):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix, dir=dir)
        filepath = temp_file.name
        temp_file.close()
        self.temp_files.add(filepath)
        return filepath
    
    def register_file(self, filepath: str):
        """Register existing file for cleanup"""
        self.temp_files.add(filepath)
    
    def cleanup_file(self, filepath: str):
        Path(filepath).unlink(missing_ok=True)
        self.temp_files.discard(filepath)
    
    def cleanup_all(self):
        for filepath in self.temp_files.copy():
            self.cleanup_file(filepath)

# Global instance
temp_manager = TempFileManager()
