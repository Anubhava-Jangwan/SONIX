"""Ring buffer for 4-second windows (stub)"""
import numpy as np

class RingBuffer:
    def __init__(self, capacity=64000):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
    
    def push(self, samples):
        pass
    
    def get_emitted_windows(self):
        return []
