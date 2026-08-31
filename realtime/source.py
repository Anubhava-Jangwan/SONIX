"""Audio source abstraction"""
import numpy as np

class SourceAdapter:
    def __init__(self, caller="unknown"):
        self.caller = caller
    
    def read(self, size):
        return None

class WavFileSource(SourceAdapter):
    def __init__(self, path):
        super().__init__("wavfile")
        self.path = path
    
    def read(self, size):
        return None
