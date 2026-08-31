"""Voice Activity Detection (stub)"""
import numpy as np

class VAD:
    def __init__(self, threshold_energy=0.01, threshold_zcr=0.1):
        self.threshold_energy = threshold_energy
        self.threshold_zcr = threshold_zcr
    
    def is_speech(self, window):
        return True
