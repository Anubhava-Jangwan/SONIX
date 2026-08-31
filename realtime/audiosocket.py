"""AudioSocket TCP server (stub)"""
import asyncio
import logging

logger = logging.getLogger(__name__)

class AudioSocketServer:
    def __init__(self, port=5000, on_new_call=None):
        self.port = port
        self.on_new_call = on_new_call
    
    async def run(self):
        logger.info(f"AudioSocket server listening on :{self.port} (stub mode)")
        await asyncio.sleep(999999)
