"""SONIX Live Detection Server - Entrypoint"""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional
import numpy as np

from aiohttp import web
from realtime.session import Session, CallState
from realtime.engine import ScoringEngine
from realtime.audiosocket import AudioSocketServer
from realtime.pairing import PairingCodeManager
from realtime.source import WavFileSource, MicSource
from realtime.miccapture import mic_page_handler
from realtime.resample import TARGET_SR, pcm16_to_float32, resample

logger = logging.getLogger(__name__)


class SonicServer:
    """Main server: orchestrates audio capture, scoring, and UI."""

    def __init__(
        self,
        port: int = 5000,
        ws_port: int = 8000,
        mock: bool = True,
        checkpoint: Optional[str] = None,
        mode: str = "voip",
        max_calls: int = 4,
        max_batch_size: int = 8,
        host: str = "0.0.0.0",
        output_dir: str = "outputs/calls"
    ):
        self.port = port
        self.ws_port = ws_port
        self.mock = mock
        self.checkpoint = checkpoint
        self.mode = mode
        self.max_calls = max_calls
        self.max_batch_size = max_batch_size
        self.host = host
        self.output_dir = output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # A missing or unloadable checkpoint must not stop the server: capture,
        # consent and the audit trail are still worth demonstrating. We fall back
        # to mock and leave scoring switched off rather than exiting.
        self.load_error = None
        try:
            self.engine = ScoringEngine(
                mock=mock,
                checkpoint_path=checkpoint,
                device=None,          # let score_file pick cuda/cpu itself
                max_batch_size=max_batch_size,
                on_broadcast=self._on_scores_ready
            )
        except Exception as e:
            if mock:
                raise
            self.load_error = str(e)
            logger.error(f"Checkpoint failed to load ({e}). Falling back to mock; "
                         "scoring stays switched off.")
            self.mock = mock = True
            self.engine = ScoringEngine(
                mock=True, on_broadcast=self._on_scores_ready)

        self.pairing_manager = PairingCodeManager(expiry_sec=120)
        self.ws_clients: Set[web.WebSocketResponse] = set()
        self.sessions: Dict[str, Session] = {}

        # Browser-mic sessions are bound to the WebSocket that carries their audio
        self.ws_calls: Dict[web.WebSocketResponse, str] = {}
        self.ws_rates: Dict[web.WebSocketResponse, int] = {}

        # Only true once a REAL trained head is loaded. The dashboard hides the
        # risk band while this is False, so we never show a number that came
        # from a mock scorer as if it meant something.
        # Derived from what actually loaded, not from which flags were passed.
        self.scoring_available = (not mock) and self.load_error is None

        # True when the loaded head is an UNTRAINED dev checkpoint. Scoring is
        # still "available" so the whole path can be exercised, but every client
        # must label the output as meaningless.
        self.scoring_synthetic = bool(
            getattr(self.engine, "config", {}).get("synthetic", False))
        if self.scoring_synthetic:
            logger.warning("SYNTHETIC checkpoint loaded - scores are NOISE. "
                           "Plumbing and latency testing only, never a demo.")

        logger.info(f"Server initialized: mode={mode}, max_calls={max_calls}, mock={mock}")

    async def _on_new_call(self, call_id: str, source):
        """Called when new call arrives."""
        if len(self.sessions) >= self.max_calls:
            logger.warning(f"Max calls reached, rejecting {call_id}")
            return False

        pairing_code = self.pairing_manager.generate()
        session = Session(
            call_id=call_id,
            source=source,
            pairing_code=pairing_code,
            pairing_expiry_sec=120
        )

        await session.request_consent()
        await self.engine.add_session(session)
        self.sessions[call_id] = session

        logger.info(f"New call: {call_id} | Code: {pairing_code}")

        await self._broadcast({
            "type": "pairing_request",
            "call_id": call_id,
            "pairing_code": pairing_code,
            "expires_in": 120,
            "caller": source.caller
        })

        return True

    async def _on_pairing_approved(self, call_id: str):
        """Called when user approves pairing code."""
        session = self.sessions.get(call_id)
        if not session:
            logger.warning(f"Pairing approval for unknown call {call_id}")
            return

        await session.on_pairing_approved()
        await self._broadcast({
            "type": "call_state", "call_id": call_id, "state": session.state.value
        })
        logger.info(f"Pairing approved: {call_id}")

    async def _on_call_ended(self, call_id: str):
        """Called when call disconnects."""
        session = self.sessions.get(call_id)
        if not session:
            return

        await session.end_call()
        session.save_audit(self.output_dir)
        await self.engine.remove_session(call_id)
        del self.sessions[call_id]

        logger.info(f"Call ended: {call_id}")

    async def _on_scores_ready(self, scores_dict: dict):
        """Called by engine when batch of scores is ready."""
        await self._broadcast({
            "type": "scores",
            "timestamp": datetime.now().isoformat(),
            "data": scores_dict
        })

    async def _broadcast(self, message: dict):
        """Send JSON to all WebSocket clients."""
        payload = json.dumps(message)
        dead_clients = set()

        for ws in self.ws_clients:
            try:
                await ws.send_str(payload)
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")
                dead_clients.add(ws)

        self.ws_clients -= dead_clients

    async def websocket_handler(self, request):
        """WebSocket endpoint for live UI."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)

        logger.info("WebSocket client connected")

        try:
            state = {
                "type": "server_state",
                "active_calls": len(self.sessions),
                "max_calls": self.max_calls,
                "engine_stats": self.engine.get_stats()
            }
            await ws.send_str(json.dumps(state))

            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)

                        if data.get("type") == "approve_pairing":
                            call_id = data.get("call_id")
                            await self._on_pairing_approved(call_id)

                        elif data.get("type") == "end_call":
                            call_id = data.get("call_id")
                            await self._on_call_ended(call_id)

                        elif data.get("type") == "start_mic_call":
                            await self._start_mic_call(ws, data)

                        elif data.get("type") == "ping":
                            await ws.send_str(json.dumps({
                                "type": "pong",
                                "timestamp": datetime.now().isoformat(),
                                "active_calls": len(self.sessions)
                            }))

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {msg.data}")

                elif msg.type == web.WSMsgType.BINARY:
                    # Raw int16 PCM from the browser microphone.
                    call_id = self.ws_calls.get(ws)
                    session = self.sessions.get(call_id) if call_id else None
                    if session is None:
                        continue

                    samples = pcm16_to_float32(msg.data)
                    sr = self.ws_rates.get(ws, TARGET_SR)
                    if sr != TARGET_SR:
                        samples = resample(samples, sr, TARGET_SR)
                    await session.push_audio(samples)

        except asyncio.CancelledError:
            logger.info("WebSocket connection cancelled")
        finally:
            self.ws_clients.discard(ws)
            call_id = self.ws_calls.pop(ws, None)
            self.ws_rates.pop(ws, None)
            if call_id and call_id in self.sessions:
                logger.info(f"Mic socket closed, ending {call_id}")
                await self._on_call_ended(call_id)

        return ws

    async def _start_mic_call(self, ws, data: dict):
        """Create a session fed by browser microphone audio over this socket."""
        if len(self.sessions) >= self.max_calls:
            await ws.send_str(json.dumps({
                "type": "error", "message": f"max concurrent calls ({self.max_calls}) reached"
            }))
            return

        call_id = f"mic_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        pairing_code = self.pairing_manager.generate()
        sample_rate = int(data.get("sample_rate") or TARGET_SR)

        source = MicSource(caller=data.get("caller", "browser-mic"), sample_rate=sample_rate)
        session = Session(call_id, source, pairing_code=pairing_code)

        # CONNECTING -> CONSENT_PENDING. Without this, on_pairing_approved() is a
        # no-op and push_audio() silently drops every chunk.
        await session.request_consent()

        await self.engine.add_session(session)
        self.sessions[call_id] = session
        self.ws_calls[ws] = call_id
        self.ws_rates[ws] = sample_rate

        await ws.send_str(json.dumps({
            "type": "mic_call_started",
            "call_id": call_id,
            "pairing_code": pairing_code,
            "sample_rate": sample_rate,
        }))
        await self._broadcast({
            "type": "pairing_request",
            "call_id": call_id,
            "pairing_code": pairing_code,
            "expires_in": 120,
            "caller": source.caller,
        })
        logger.info(f"Mic call {call_id} started @ {sample_rate} Hz, code {pairing_code}")

    async def http_approve_handler(self, request):
        """Approve a pairing code from the dashboard (HTTP, so Streamlit can call it)."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "expected JSON body"}, status=400)

        call_id = data.get("call_id")
        session = self.sessions.get(call_id)
        if session is None:
            return web.json_response({"error": f"unknown call {call_id}"}, status=404)

        await self._on_pairing_approved(call_id)
        return web.json_response({"call_id": call_id, "state": session.state.value})

    async def http_end_call_handler(self, request):
        """End a call from the dashboard."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "expected JSON body"}, status=400)

        call_id = data.get("call_id")
        if call_id not in self.sessions:
            return web.json_response({"error": f"unknown call {call_id}"}, status=404)

        await self._on_call_ended(call_id)
        return web.json_response({"call_id": call_id, "state": "ended"})

    async def http_telemetry_handler(self, request):
        """Per-call window telemetry for the dashboard chart."""
        wanted = request.query.get("call_id")
        limit = int(request.query.get("limit", 240))

        calls = {
            cid: s.telemetry(limit=limit)
            for cid, s in self.sessions.items()
            if not wanted or cid == wanted
        }
        return web.json_response({
            "scoring_available": self.scoring_available,
            "scoring_synthetic": self.scoring_synthetic,
            "mode": self.mode,
            "active_calls": len(self.sessions),
            "max_calls": self.max_calls,
            "engine_stats": self.engine.get_stats(),
            "calls": calls,
            "timestamp": datetime.now().isoformat(),
        })

    async def http_upload_handler(self, request):
        """Handle WAV file uploads for post-call scoring."""
        try:
            data = await request.post()
            file_field = data.get('file')

            if not file_field:
                return web.json_response({"error": "No file provided"}, status=400)

            call_id = f"upload_{datetime.now().isoformat().replace(':', '-')}"
            logger.info(f"Processing upload: {call_id}")

            temp_path = f"/tmp/{call_id}.wav"
            with open(temp_path, 'wb') as f:
                f.write(file_field.file.read())

            source = WavFileSource(temp_path)
            session = Session(call_id, source, pairing_code="upload_mode", pairing_expiry_sec=1)

            await session.request_consent()
            await session.on_pairing_approved()
            await self.engine.add_session(session)
            self.sessions[call_id] = session

            while True:
                samples = source.read(8000)
                if samples is None or len(samples) == 0:
                    break
                await session.push_audio(samples)
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.5)
            await self._on_call_ended(call_id)

            scores_list = [s["score"] for s in session.scores.values()]
            response = {
                "call_id": call_id,
                "status": "success",
                "windows_scored": len(session.scores),
                "summary": {
                    "mean_score": float(np.mean(scores_list)) if scores_list else None,
                    "max_score": float(np.max(scores_list)) if scores_list else None,
                    "min_score": float(np.min(scores_list)) if scores_list else None
                },
                "scores": session.scores
            }

            return web.json_response(response)

        except Exception as e:
            logger.error(f"Upload handler error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def http_status_handler(self, request):
        """Return server status."""
        return web.json_response({
            "status": "ok",
            "mode": self.mode,
            "active_calls": len(self.sessions),
            "max_calls": self.max_calls,
            "engine_stats": self.engine.get_stats(),
            "ws_clients": len(self.ws_clients),
            "timestamp": datetime.now().isoformat()
        })

    async def run(self):
        """Start all server components."""
        logger.info("=== SONIX Server Starting ===")

        engine_task = asyncio.create_task(self.engine.run())
        logger.info(f"Engine started (mock={self.mock})")

        if self.mode == "voip":
            audio_server = AudioSocketServer(
                port=self.port,
                on_new_call=self._on_new_call
            )
            audio_task = asyncio.create_task(audio_server.run())
            logger.info(f"AudioSocket server started on :{self.port}")

        app = web.Application()
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_post('/api/score-file', self.http_upload_handler)
        app.router.add_get('/api/status', self.http_status_handler)
        app.router.add_get('/api/telemetry', self.http_telemetry_handler)
        app.router.add_post('/api/approve', self.http_approve_handler)
        app.router.add_post('/api/end-call', self.http_end_call_handler)
        app.router.add_get('/mic', mic_page_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.ws_port)
        await site.start()

        logger.info(f"WebSocket server started on ws://{self.host}:{self.ws_port}")
        logger.info(f"Microphone capture page: http://localhost:{self.ws_port}/mic")
        if not self.scoring_available:
            logger.warning("SCORING DISABLED - no real checkpoint loaded. "
                           "Audio path runs; risk band stays hidden.")
        logger.info("=== SONIX Server Ready ===")

        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Server stopping...")
            engine_task.cancel()
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="SONIX Live Detection Server")
    parser.add_argument('--port', type=int, default=5000, help="AudioSocket TCP port")
    parser.add_argument('--ws-port', type=int, default=8000, help="WebSocket+HTTP port")
    parser.add_argument('--mock', action='store_true', default=False,
                        help="Use the mock scorer (no checkpoint needed)")
    parser.add_argument('--ckpt', type=str, default=None, help="Path to head.pt")
    parser.add_argument('--mode', choices=['voip', 'webrtc', 'upload'], default='voip')
    parser.add_argument('--max-calls', type=int, default=4)
    parser.add_argument('--max-batch-size', type=int, default=8,
                        help="Windows scored per forward pass. Measure with "
                             "realtime/selftest.py: on a GTX 1650, 8 overruns "
                             "the 0.5s hop (744ms) but 4 fits (394ms).")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--output-dir', type=str, default='outputs/calls')
    parser.add_argument('--log-level', default='INFO')

    args = parser.parse_args()

    # Previously --mock defaulted to True, so real scoring was unreachable even
    # with --ckpt. Now the checkpoint decides, and we say which one is in force.
    use_mock = args.mock or args.ckpt is None
    if use_mock and not args.mock:
        print("[SONIX] No --ckpt given; falling back to the mock scorer. "
              "Risk band will stay hidden in the dashboard.")

    logging.basicConfig(
        level=args.log_level,
        format='[%(asctime)s] %(name)s:%(levelname)s - %(message)s'
    )

    server = SonicServer(
        port=args.port,
        ws_port=args.ws_port,
        mock=use_mock,
        checkpoint=args.ckpt,
        mode=args.mode,
        max_calls=args.max_calls,
        max_batch_size=args.max_batch_size,
        host=args.host,
        output_dir=args.output_dir
    )

    asyncio.run(server.run())


if __name__ == '__main__':
    main()
