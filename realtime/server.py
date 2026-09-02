"""SONIX Live Detection Server - Entrypoint"""

import asyncio
import argparse
import logging
import json
import tempfile
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
from realtime import models as model_registry

logger = logging.getLogger(__name__)

# aiohttp defaults to a 1 MB request body, which rejects essentially every real
# call recording with a bare "Content Too Large". 256 MB is far more than any
# clip we demo and still bounded.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024


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
        host: str = "0.0.0.0",
        output_dir: str = "outputs/calls"
    ):
        self.port = port
        self.ws_port = ws_port
        self.mock = mock
        self.checkpoint = checkpoint
        self.mode = mode
        self.max_calls = max_calls
        self.host = host
        self.output_dir = output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.engine = ScoringEngine(
            mock=mock,
            checkpoint_path=checkpoint,
            device="cuda",
            on_broadcast=self._on_scores_ready
        )

        self.pairing_manager = PairingCodeManager(expiry_sec=120)
        self.ws_clients: Set[web.WebSocketResponse] = set()
        self.sessions: Dict[str, Session] = {}

        # Browser-mic sessions are bound to the WebSocket that carries their audio
        self.ws_calls: Dict[web.WebSocketResponse, str] = {}
        self.ws_rates: Dict[web.WebSocketResponse, int] = {}

        # Only true once a REAL trained head is loaded. The dashboard hides the
        # risk band while this is False, so we never show a number that came
        # from a mock scorer as if it meant something.
        self.scoring_available = (not mock) and checkpoint is not None

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
        session = Session(call_id, source, pairing_code=pairing_code,
                          vad_energy=getattr(self, 'vad_energy', None))

        # CONNECTING -> CONSENT_PENDING. Without this, on_pairing_approved() is a
        # no-op and push_audio() silently drops every chunk.
        await session.request_consent()

        # The engine only scores sessions in LISTENING/SCORING. Until pairing is
        # approved nothing is scored and the dashboard looks silently broken --
        # which is exactly what it did. --auto-approve skips that for demos.
        if getattr(self, 'auto_approve', False):
            await session.on_pairing_approved()
            logger.info(f"[{call_id}] auto-approved (--auto-approve): scoring now")

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
            "warm": bool(getattr(self.engine, "warm", True)),
            "warming": bool(getattr(self.engine, "warming", False)),
            "models": self.engine.model_catalogue() if not self.mock else [],
            "default_model": self.engine.default_key if not self.mock else None,
            "mode": self.mode,
            "active_calls": len(self.sessions),
            "max_calls": self.max_calls,
            "engine_stats": self.engine.get_stats(),
            "calls": calls,
            "timestamp": datetime.now().isoformat(),
        })

    async def http_models_handler(self, request):
        """Which trained heads this server can score with, and which exist on disk."""
        return web.json_response({
            "mock": self.mock,
            "warm": bool(getattr(self.engine, "warm", True)),
            "warming": bool(getattr(self.engine, "warming", False)),
            "default": self.engine.default_key if not self.mock else None,
            "models": self.engine.model_catalogue() if not self.mock else [],
        })

    @staticmethod
    def _adaptive_vad_floor(samples, frame: int = 400):
        """Pick a silence-gate energy floor from the clip's own speech level.

        The gate's default floor is a FIXED 0.01 RMS (~-40 dBFS per 25 ms
        frame), tuned for studio-level speech. A phone recording, a quiet room
        or a distant mic sits entirely below it, so every window is thrown away
        as "silence" and the upload scores nothing at all -- which is what the
        dashboard was reporting. Scaling the floor to the clip keeps genuine
        digital silence out while letting quiet speech through.

        Returns (floor, speech_rms, floor_dbfs) or None if the clip is empty.
        """
        w = np.asarray(samples, dtype=np.float32).reshape(-1)
        if w.size < frame:
            return None
        nf = w.size // frame
        rms = np.sqrt(np.mean(np.square(w[:nf * frame].reshape(nf, frame).astype(np.float64)), axis=1))
        speech = float(np.percentile(rms, 90))          # a loud frame, not the peak
        floor = float(np.clip(speech * 0.12, 0.0006, 0.01))
        return floor, speech, 20.0 * np.log10(max(floor, 1e-12))

    async def _feed_upload(self, call_id: str, session, source, chunk: int = 8000,
                           pace: float = 0.02):
        """Push an uploaded file into a session in the background.

        This used to run inline inside the request, so the browser sat on a
        blocked POST for the whole clip and every score arrived at once at the
        end. Feeding in a task means the dashboard can poll /api/telemetry and
        watch the risk line build window by window, which is the whole point of
        a live-audio demo.
        """
        try:
            while True:
                samples = source.read(chunk)
                if samples is None or len(samples) == 0:
                    break
                await session.push_audio(samples)
                await asyncio.sleep(pace)
        except Exception as exc:
            logger.error(f"[{call_id}] upload feed failed: {exc}", exc_info=True)
        finally:
            session.feed_done = True
            logger.info(f"[{call_id}] upload feed finished "
                        f"({session.ringbuffer.windows_emitted} windows emitted)")

    async def http_upload_handler(self, request):
        """Accept a file, start scoring it, and return immediately.

        The response carries the call_id and how many windows to expect; the
        dashboard then polls /api/telemetry?call_id=... and draws the timeline
        as the scores land. Pass wait=1 to get the old blocking behaviour back
        (used by scripts that just want the final numbers).
        """
        try:
            data = await request.post()
            file_field = data.get('file')

            if not file_field:
                return web.json_response({"error": "No file provided"}, status=400)

            gate = (data.get('vad') or 'auto').strip().lower()
            requested = (data.get('model') or '').strip() or None
            if requested and not self.mock:
                try:
                    await asyncio.to_thread(self.engine.ensure_model, requested)
                except Exception as exc:
                    return web.json_response(
                        {"error": f"model '{requested}' unavailable: {exc}"}, status=400)

            wait = str(data.get('wait') or '').lower() in ('1', 'true', 'yes')

            call_id = f"upload_{datetime.now().strftime('%Y%m%dT%H%M%S_%f')}"
            model_key = requested or (self.engine.default_key if not self.mock else None)
            logger.info(f"Processing upload: {call_id} (model={model_key})")

            # /tmp does not exist on Windows -- this used to fail outright there.
            suffix = Path(getattr(file_field, "filename", "") or "upload.wav").suffix or ".wav"
            fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                             prefix=f"sonix_{call_id}_")
            fd.write(file_field.file.read())
            fd.close()
            temp_path = fd.name

            source = WavFileSource(temp_path)
            try:
                # Decoding + resampling is CPU work; a long clip would otherwise
                # stall every other request for its duration.
                duration = await asyncio.to_thread(lambda: source.duration_sec)
            except Exception as exc:
                return web.json_response({"error": f"could not decode audio: {exc}"},
                                         status=400)

            n = int(round(duration * TARGET_SR))
            win, hop = 64000, 8000
            if n < win:
                # A clip shorter than one window emits NOTHING in the live path.
                # Repeat-pad it rather than zero-pad: padding with digital
                # silence is exactly the bug that made short real clips score
                # 0.88 ("fake") instead of 0.03 ("real").
                source._load()
                reps = int(np.ceil(win / max(1, source._samples.size)))
                source._samples = np.tile(source._samples, reps)[:win].astype(np.float32)
                source._pos = 0
                n = win
                logger.info(f"[{call_id}] clip is {duration:.2f}s (< 4s); "
                            f"repeat-padded to one full window")
            expected_windows = (n - win) // hop + 1

            # Silence gate: "off" scores every window, "strict" keeps the
            # fixed studio-level default, "auto" (the default) scales the floor
            # to this clip so a quiet recording is not discarded wholesale.
            vad_floor = getattr(self, 'vad_energy', None)
            vad_note = "server default"
            if gate == "off":
                vad_floor, vad_note = 0.0, "disabled - every window scored"
            elif gate != "strict" and vad_floor is None:
                source._load()
                adapt = self._adaptive_vad_floor(source._samples)
                if adapt:
                    vad_floor, speech, floor_db = adapt
                    vad_note = (f"auto {vad_floor:.5f} ({floor_db:.1f} dBFS) "
                                f"from speech level {speech:.5f}")

            session = Session(call_id, source, pairing_code="upload_mode",
                              pairing_expiry_sec=1,
                              vad_energy=vad_floor,
                              model_key=model_key)
            session.expected_windows = int(expected_windows)
            logger.info(f"[{call_id}] silence gate: {vad_note}")

            await session.request_consent()
            await session.on_pairing_approved()
            await self.engine.add_session(session)
            self.sessions[call_id] = session

            feed = asyncio.create_task(self._feed_upload(call_id, session, source))

            if not wait:
                return web.json_response({
                    "call_id": call_id,
                    "status": "streaming",
                    "model": model_key,
                    "duration_s": round(float(duration), 3),
                    "expected_windows": int(expected_windows),
                    "vad": vad_note,
                    "poll": f"/api/telemetry?call_id={call_id}",
                })

            # Blocking path: wait for the feed, then for scoring to drain.
            await feed
            for _ in range(600):                       # 60s ceiling
                if len(session.scores) >= expected_windows or not session.pending_windows:
                    if len(session.scores) >= expected_windows:
                        break
                await asyncio.sleep(0.1)
            await self._on_call_ended(call_id)

            scores_list = [s["score"] for s in session.scores.values()]
            return web.json_response({
                "call_id": call_id,
                "status": "success",
                "model": model_key,
                "windows_scored": len(session.scores),
                "expected_windows": int(expected_windows),
                "summary": {
                    "mean_score": float(np.mean(scores_list)) if scores_list else None,
                    "max_score": float(np.max(scores_list)) if scores_list else None,
                    "min_score": float(np.min(scores_list)) if scores_list else None
                },
                "scores": session.scores
            })

        except (web.HTTPRequestEntityTooLarge, ValueError) as e:
            # aiohttp raises this while parsing an over-size body in .post().
            logger.error(f"Upload too large: {e}")
            return web.json_response(
                {"error": f"file exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB "
                          f"upload limit"}, status=413)
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

        # Pay the model-loading cost now, in a thread, instead of inside the
        # first upload -- where it looked exactly like a hung server.
        warm_task = asyncio.create_task(self.engine.preload())

        if self.mode == "voip":
            audio_server = AudioSocketServer(
                port=self.port,
                on_new_call=self._on_new_call
            )
            audio_task = asyncio.create_task(audio_server.run())
            logger.info(f"AudioSocket server started on :{self.port}")

        app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_post('/api/score-file', self.http_upload_handler)
        app.router.add_get('/api/status', self.http_status_handler)
        app.router.add_get('/api/models', self.http_models_handler)
        app.router.add_get('/api/telemetry', self.http_telemetry_handler)
        app.router.add_post('/api/approve', self.http_approve_handler)
        app.router.add_post('/api/end-call', self.http_end_call_handler)
        app.router.add_get('/mic', mic_page_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.ws_port)
        try:
            await site.start()
        except OSError as exc:
            # A stale server from an earlier run holding the port is the single
            # most common way this fails, and the raw 30-line traceback buries
            # that. Say it plainly, with the command that fixes it.
            print()
            print("=" * 70)
            print(f"PORT {self.ws_port} IS ALREADY IN USE")
            print("=" * 70)
            print("Another SONIX server is almost certainly still running from")
            print("an earlier run -- and it is running the OLD code, so anything")
            print("you test against it will behave like the old build.")
            print()
            print("Kill it, then start this one again:")
            print()
            print("    taskkill /F /IM python.exe          (Windows)")
            print("    pkill -f realtime.server            (macOS/Linux)")
            print()
            print(f"Or run this server on a free port:  --ws-port {self.ws_port + 1}")
            print("=" * 70)
            engine_task.cancel()
            warm_task.cancel()
            raise SystemExit(1) from exc

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
    parser.add_argument('--auto-approve', action='store_true', default=False,
                        help="skip the pairing/consent step and start scoring "
                             "immediately. For demos and testing only -- the "
                             "consent gate exists for a reason in real use.")
    parser.add_argument('--vad-energy', type=float, default=None,
                        help="VAD energy floor (default 0.01 ~= -40 dBFS). "
                             "Lower it (e.g. 0.003) if a quiet mic is being "
                             "rejected as silence and nothing gets scored.")
    parser.add_argument('--max-calls', type=int, default=4)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--output-dir', type=str, default='outputs/calls')
    parser.add_argument('--log-level', default='INFO')

    args = parser.parse_args()

    # Previously --mock defaulted to True, so real scoring was unreachable even
    # with --ckpt. Now the checkpoint decides, and we say which one is in force.
    # And if --ckpt is omitted we look for the known heads ourselves rather than
    # silently dropping to mock -- having to name a path to get real scoring was
    # the single most common way this server came up useless.
    ckpt = args.ckpt
    catalogue = model_registry.catalogue()
    present = [m for m in catalogue if m["exists"]]
    if ckpt is None and not args.mock and present:
        default = next((m for m in present if m["key"] == model_registry.DEFAULT_KEY),
                       present[0])
        ckpt = default["resolved_path"]
        print(f"[SONIX] No --ckpt given; using {default['label']} "
              f"({default['path']}).")

    use_mock = args.mock or ckpt is None
    if use_mock and not args.mock:
        print("[SONIX] No trained head found under outputs/models/; falling back "
              "to the mock scorer. Risk band will stay hidden in the dashboard.")
    if not use_mock:
        found = ", ".join(m["label"] for m in present) or "none"
        missing = ", ".join(m["label"] for m in catalogue if not m["exists"])
        print(f"[SONIX] Heads available to the dashboard: {found}")
        if missing:
            print(f"[SONIX] Not on disk (tab will be disabled): {missing}")

    logging.basicConfig(
        level=args.log_level,
        format='[%(asctime)s] %(name)s:%(levelname)s - %(message)s'
    )

    server = SonicServer(
        port=args.port,
        ws_port=args.ws_port,
        mock=use_mock,
        checkpoint=ckpt,
        mode=args.mode,
        max_calls=args.max_calls,
        host=args.host,
        output_dir=args.output_dir
    )
    server.auto_approve = args.auto_approve
    server.vad_energy = args.vad_energy

    if args.auto_approve:
        print("[SONIX] --auto-approve: calls start scoring immediately, no "
              "pairing step. Demo/testing only.")
    else:
        print("[SONIX] Calls wait for pairing approval before scoring. "
              "Approve in the dashboard, POST /api/approve, or use "
              "--auto-approve.")
    if args.vad_energy is not None:
        print(f"[SONIX] VAD energy floor set to {args.vad_energy} "
              f"(default 0.01).")

    asyncio.run(server.run())


if __name__ == '__main__':
    main()
