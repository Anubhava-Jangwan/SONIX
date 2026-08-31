"""Mock scorer (deterministic, no torch)"""
import numpy as np

class MockScorer:
    def score(self, embeddings):
        batch_size = len(embeddings)
        return np.random.RandomState(42).rand(batch_size).astype(np.float32) * 0.5
