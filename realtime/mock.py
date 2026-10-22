"""Mock scorer (deterministic, no torch)"""
import numpy as np

class MockScorer:
    def score(self, windows):
        batch_size = len(windows)
        return np.random.RandomState(42).rand(batch_size).astype(np.float32) * 0.5
