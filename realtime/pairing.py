"""Device pairing with 6-digit codes"""
import random

class PairingCodeManager:
    def __init__(self, expiry_sec=120):
        self.expiry_sec = expiry_sec
    
    def generate(self):
        return f"{random.randint(100000, 999999)}"
