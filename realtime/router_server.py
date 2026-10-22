"""SONIX Multi-Head Console - server.

Scores one uploaded clip with every trained head at once and shows where they
agree or disagree, instead of hiding that behind a single number. Standalone:
own port (8100), own page, touches nothing else in the repo.

THE ONE THING THIS FILE MUST GET RIGHT: the 9 heads in outputs/models/ are a
nested ladder (each one's training data is a superset of the previous one's,
see realtime/models.py), not specialists picking different jobs. So the gate
below NEVER decides which head's verdict counts -- the verdict is max() across
every head, a safety floor. The gate only produces soft weights plus a
one-line explanation of which head is most relevant and why, clearly labelled
"heuristic" so the UI can never be mistaken for a learned router that doesn't
exist.

Reuses, rather than reimplements:
  realtime/models.py      - the head registry (key, label, training note, path)
  realtime/engine.py      - ScoringEngine: preload(), _embed_windows() (the
                             frontend runs ONCE), _score_windows()/ensure_model()
                             (looped once per head on that same embedding)
  realtime/ringbuffer.py  - RingBuffer: 4.0s window / 0.5s hop windowing
  realtime/vad.py         - VAD: the silence gate, and its per-window RMS is
                             reused as the input to the SNR heuristic below
  realtime/source.py      - WavFileSource: decode + resample an upload
  realtime/server.py      - _origin_allowed() / MAX_UPLOAD_BYTES (imported,
                             not copied) so this endpoint has the same
                             localhost-only posture as the existing one
"""

import argparse
import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from aiohttp import web

from realtime import models as model_registry
from realtime.engine import ScoringEngine
from realtime.ringbuffer import RingBuffer
from realtime.server import MAX_UPLOAD_BYTES, _origin_allowed
from realtime.source import WavFileSource
from realtime.vad import VAD

logger = logging.getLogger(__name__)

TARGET_SR = 16000
PAGE_PATH = Path(__file__).resolve().parent / "router_page.html"

# Same provisional thresholds realtime/live_ui.py and the /mic page already
# use. There is no single shared constant for this in the repo -- every
# surface hardcodes its own -- so this matches the existing convention rather
# than inventing a fourth one.
AMBER_AT = 0.35
RED_AT = 0.65

# Windows per front-end forward pass. Embedding an entire multi-minute clip in
# one batch is what OOMs a 4GB GPU (this is an offline console, not the
# latency-bound live path, so unlike realtime/server.py's --max-batch-size
# this number is chosen for memory headroom, not hop-budget timing).
EMBED_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# The heuristic gate. Transparent on purpose: one readable function, named
# thresholds, and it only ever nudges weights -- never the verdict.
# ---------------------------------------------------------------------------

# Which head families were trained on which augmentation, read straight off
# the descriptions in realtime/models.py. Kept here as structured data since
# the registry's notes are free text, not meant to be parsed.
HEAD_TRAITS = {
    "baseline":          {"codec_trained": False, "noise_trained": False, "indic_trained": False},
    "baseline_rebuilt":  {"codec_trained": False, "noise_trained": False, "indic_trained": False},
    "augmented":         {"codec_trained": True,  "noise_trained": False, "indic_trained": False},
    "augmented_rebuilt": {"codec_trained": True,  "noise_trained": False, "indic_trained": False},
    "robust":            {"codec_trained": True,  "noise_trained": False, "indic_trained": False},
    "robust_v2":         {"codec_trained": True,  "noise_trained": True,  "indic_trained": False},
    "full_ho":           {"codec_trained": True,  "noise_trained": True,  "indic_trained": True},
    "full_indic_as5":    {"codec_trained": True,  "noise_trained": True,  "indic_trained": True},
    "v3":                {"codec_trained": True,  "noise_trained": True,  "indic_trained": True},
}

NARROWBAND_HZ = 4000.0     # below this, treat the channel as telephony
LOW_SNR_DB = 15.0          # below this, treat the clip as noisy/reverberant
CODEC_BOOST = 1.5
NOISE_BOOST = 1.5
BANDWIDTH_FFT_SECONDS = 8.0
BANDWIDTH_ENERGY_FRACTION = 0.99


def _estimate_bandwidth_hz(samples, sr=TARGET_SR):
    """Frequency below which 99% of spectral energy sits, over up to the
    first 8s of the clip. Narrowband/telephony audio (G.711's ~3.4kHz cutoff)
    scores low; clean wideband speech sits near Nyquist (8kHz @ 16kHz)."""
    n = min(len(samples), int(sr * BANDWIDTH_FFT_SECONDS))
    if n < 512:
        return float(sr) / 2.0
    x = np.asarray(samples[:n], dtype=np.float64) * np.hanning(n)
    power = np.abs(np.fft.rfft(x)) ** 2
    cum = np.cumsum(power)
    total = cum[-1] if cum[-1] > 0 else 1.0
    idx = int(np.searchsorted(cum, BANDWIDTH_ENERGY_FRACTION * total))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return float(freqs[min(idx, len(freqs) - 1)])


def _estimate_snr_db(window_stats):
    """Mean RMS of VAD-passed windows vs VAD-rejected windows, in dB. Reuses
    the RMS values VAD.is_speech() already computed -- no second pass over
    the audio. Falls back to the quietest fifth of windows as a noise-floor
    proxy when nothing was gated out, rather than claiming an SNR with no
    evidence behind it."""
    speech = [s["rms"] for s in window_stats if s.get("passed") and s.get("rms", 0) > 0]
    noise = [s["rms"] for s in window_stats if not s.get("passed") and s.get("rms", 0) > 0]
    if not noise:
        ordered = sorted(s["rms"] for s in window_stats if s.get("rms", 0) > 0)
        cut = max(1, len(ordered) // 5)
        noise = ordered[:cut]
    if not speech or not noise:
        return 40.0  # nothing to compare against -- assume clean
    speech_db = 20.0 * np.log10(np.mean(speech) + 1e-9)
    noise_db = 20.0 * np.log10(np.mean(noise) + 1e-9)
    return float(np.clip(speech_db - noise_db, -10.0, 60.0))


def compute_gate(bandwidth_hz, snr_db, duration_s, head_keys):
    """The heuristic gate: soft weights + a plain-language explanation. Never
    picks a winner -- see the module docstring. `duration_s` is reported for
    context only and never biases a weight: no head in the registry is
    duration-specialised, so inventing that correlation would be exactly the
    kind of fake router this app exists to NOT be."""
    narrowband = bandwidth_hz < NARROWBAND_HZ
    low_snr = snr_db < LOW_SNR_DB

    weights = {}
    for key in head_keys:
        traits = HEAD_TRAITS.get(key, {})
        w = 1.0
        if narrowband and traits.get("codec_trained"):
            w *= CODEC_BOOST
        if low_snr and traits.get("noise_trained"):
            w *= NOISE_BOOST
        weights[key] = w
    total = sum(weights.values()) or 1.0
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    reasons = []
    if narrowband:
        reasons.append(f"Narrowband ({bandwidth_hz:.0f} Hz) — weighting "
                       f"codec-trained heads higher")
    if low_snr:
        reasons.append(f"Low SNR ({snr_db:.1f} dB) — weighting "
                       f"reverb/noise-trained heads higher")
    if not reasons:
        reasons.append(f"Clean wideband signal ({bandwidth_hz:.0f} Hz, "
                       f"{snr_db:.1f} dB SNR) — no head family favoured")

    return {
        "kind": "heuristic",
        "explanation": "; ".join(reasons),
        "weights": weights,
        "features": {
            "bandwidth_hz": round(bandwidth_hz, 1),
            "snr_db": round(snr_db, 1),
            "duration_s": round(duration_s, 2),
        },
    }


def _band_for(score):
    if score is None:
        return "N/A"
    if score >= RED_AT:
        return "RED"
    if score >= AMBER_AT:
        return "AMBER"
    return "GREEN"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class RouterConsole:
    def __init__(self, engine, host="0.0.0.0", port=8100):
        self.engine = engine          # None when no checkpoint exists on disk
        self.host = host
        self.port = port

    # ---- handlers -----------------------------------------------------

    async def http_index(self, request):
        return web.FileResponse(PAGE_PATH)

    async def http_heads(self, request):
        catalogue = (self.engine.model_catalogue() if self.engine is not None
                    else model_registry.catalogue())
        heads = [dict(item, traits=HEAD_TRAITS.get(item["key"], {})) for item in catalogue]
        return web.json_response({"heads": heads})

    async def http_health(self, request):
        if self.engine is None:
            return web.json_response({
                "device": None, "front_end_warm": False, "front_end_warming": False,
                "heads_loaded": [], "heads_total": len(model_registry.catalogue()),
            })
        stats = self.engine.get_stats()
        return web.json_response({
            "device": self.engine.device or "auto",
            "front_end_warm": stats["warm"],
            "front_end_warming": stats["warming"],
            "heads_loaded": stats["loaded_models"],
            "heads_total": len(model_registry.catalogue()),
        })

    async def http_analyze(self, request):
        if not _origin_allowed(request, self.port):
            return web.json_response({"error": "origin not allowed"}, status=403)
        if self.engine is None:
            return web.json_response(
                {"error": "No trained head checkpoints found under outputs/models/. "
                          "Nothing to score with."}, status=400)

        temp_path = None
        try:
            data = await request.post()
            file_field = data.get("file")
            if not file_field:
                return web.json_response({"error": "No file provided"}, status=400)

            suffix = Path(getattr(file_field, "filename", "") or "upload.wav").suffix or ".wav"
            fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="sonix_router_")
            fd.write(file_field.file.read())
            fd.close()
            temp_path = fd.name

            source = WavFileSource(temp_path)
            try:
                duration = await asyncio.to_thread(lambda: source.duration_sec)
            except Exception as exc:
                return web.json_response({"error": f"could not decode audio: {exc}"}, status=400)

            win, hop = 64000, 8000
            n = int(round(duration * TARGET_SR))
            if n < win:
                # Never zero-pad a short clip -- that is the documented bug
                # that made short real clips score "fake". Repeat instead,
                # same idiom realtime/server.py's upload handler uses.
                source._load()
                reps = int(np.ceil(win / max(1, source._samples.size)))
                source._samples = np.tile(source._samples, reps)[:win].astype(np.float32)

            source._load()
            rb = RingBuffer(window_samples=win, hop_samples=hop)
            rb.push(source._samples)
            windows = rb.get_emitted_windows()
            windows_total = len(windows)

            call_id = f"analyze_{datetime.now().strftime('%Y%m%dT%H%M%S_%f')}"

            if windows_total == 0:
                return web.json_response({
                    "call_id": call_id, "duration_s": round(float(duration), 3),
                    "windows_total": 0, "windows_gated": 0,
                    "verdict": {"score": None, "band": "N/A", "rule": "max across heads"},
                    "gate": None, "heads": [],
                    "message": "Clip decoded to fewer than 4 seconds even after "
                               "repeat-padding; nothing to window.",
                }, status=200)

            vad = VAD()
            gated_windows, window_stats = [], []
            for w in windows:
                passed = vad.is_speech(w)
                window_stats.append(dict(vad.last_stats))
                if passed:
                    gated_windows.append(w)
            windows_gated = windows_total - len(gated_windows)

            if not gated_windows:
                return web.json_response({
                    "call_id": call_id, "duration_s": round(float(duration), 3),
                    "windows_total": windows_total, "windows_gated": windows_gated,
                    "verdict": {"score": None, "band": "N/A", "rule": "max across heads"},
                    "gate": None, "heads": [],
                    "message": "Every window was dropped by the silence gate -- "
                               "this clip is all silence, nothing to score.",
                }, status=200)

            # The 300M-param front-end runs ONCE PER WINDOW here -- every head
            # below scores these same embeddings, that's the whole point of the
            # app -- but a real clip can be minutes long (hundreds of windows),
            # and embedding them all in a single forward pass is what was
            # blowing out a 4GB GPU (torch.OutOfMemoryError on anything past a
            # short demo clip). Chunk it, same batch size realtime/server.py
            # defaults to, and concatenate -- the front-end still runs once
            # per window, just not in one giant batch.
            embed_chunks = []
            for i in range(0, len(gated_windows), EMBED_BATCH_SIZE):
                chunk = gated_windows[i:i + EMBED_BATCH_SIZE]
                embed_chunks.append(await self.engine._embed_windows(chunk))
            embeddings = np.concatenate(embed_chunks, axis=0)

            heads_out = []
            scored_keys = []
            for item in model_registry.catalogue():
                if not item["exists"]:
                    continue
                key = item["key"]
                try:
                    scores = await self.engine._score_windows(
                        embeddings, model_keys=[key] * len(gated_windows))
                except Exception as exc:
                    logger.warning("router: head '%s' unavailable (%s)", key, exc)
                    continue
                scores = [float(s) for s in scores]
                heads_out.append({
                    "key": key, "label": item["label"], "note": item["note"],
                    "traits": HEAD_TRAITS.get(key, {}),
                    "mean": round(float(np.mean(scores)), 4),
                    "max": round(float(np.max(scores)), 4),
                    "scores": [round(s, 4) for s in scores],
                })
                scored_keys.append(key)

            heads_out.sort(key=lambda h: h["max"], reverse=True)

            verdict_score = max((h["max"] for h in heads_out), default=None)
            verdict = {"score": verdict_score, "band": _band_for(verdict_score),
                      "rule": "max across heads"}

            bandwidth_hz = await asyncio.to_thread(_estimate_bandwidth_hz, source._samples)
            snr_db = _estimate_snr_db(window_stats)
            gate = compute_gate(bandwidth_hz, snr_db, duration, scored_keys)

            return web.json_response({
                "call_id": call_id,
                "duration_s": round(float(duration), 3),
                "windows_total": windows_total,
                "windows_gated": windows_gated,
                "verdict": verdict,
                "gate": gate,
                "heads": heads_out,
            })
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    # ---- app / run ------------------------------------------------------

    def build_app(self):
        app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
        app.router.add_get("/", self.http_index)
        app.router.add_get("/api/heads", self.http_heads)
        app.router.add_get("/api/health", self.http_health)
        app.router.add_post("/api/analyze", self.http_analyze)
        return app

    async def run(self):
        logger.info("=== SONIX Multi-Head Console Starting ===")
        if self.engine is not None:
            asyncio.create_task(self.engine.preload())

        app = self.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        try:
            await site.start()
        except OSError as exc:
            print()
            print("=" * 70)
            print(f"PORT {self.port} IS ALREADY IN USE")
            print("=" * 70)
            print("Another instance of the console is almost certainly still")
            print("running from an earlier run.")
            print()
            print("Kill it, then start this one again, or run it on a free port:")
            print(f"    python -m realtime.router_server --port {self.port + 1}")
            print("=" * 70)
            raise SystemExit(1) from exc

        logger.info(f"Console ready at http://localhost:{self.port}/")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Console stopping...")
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="SONIX Multi-Head Console")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s] %(name)s:%(levelname)s - %(message)s",
    )

    catalogue = model_registry.catalogue()
    present = [m for m in catalogue if m["exists"]]
    if not present:
        print("[SONIX] No trained head checkpoints found under outputs/models/. "
              "The console will start but /api/analyze will refuse to score "
              "anything until at least one checkpoint exists.")
        engine = None
    else:
        default = next((m for m in present if m["key"] == model_registry.DEFAULT_KEY),
                       present[0])
        found = ", ".join(m["label"] for m in present)
        missing = ", ".join(m["label"] for m in catalogue if not m["exists"])
        print(f"[SONIX] Heads available: {found}")
        if missing:
            print(f"[SONIX] Not on disk (skipped): {missing}")
        engine = ScoringEngine(mock=False, checkpoint_path=default["resolved_path"],
                               device=None)

    console = RouterConsole(engine, host=args.host, port=args.port)
    asyncio.run(console.run())


if __name__ == "__main__":
    main()
