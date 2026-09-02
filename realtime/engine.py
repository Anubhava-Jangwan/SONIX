"""Real-time scoring engine: model owner + batch inference.

Holds ONE frozen wav2vec2 front-end (it is ~300M parameters -- loading a second
copy per model would not fit) and any number of small trained heads on top, so
the same audio can be scored by baseline / augmented / robust without
restarting the server. Each session names the head it wants; the engine groups a
batch by head before running it.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch

from realtime import models as model_registry

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Single asyncio task that batches windows across concurrent calls."""

    def __init__(
        self,
        mock: bool = True,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_interval: float = 0.5,
        max_batch_size: int = 8,
        on_broadcast: Optional[Callable] = None
    ):
        self.mock = mock
        self.device = device
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.on_broadcast = on_broadcast
        self.sessions: Dict[str, object] = {}

        # key -> loaded StandardisedHead. Populated lazily by ensure_model().
        self.heads: Dict[str, object] = {}
        self.head_configs: Dict[str, dict] = {}
        self.load_errors: Dict[str, str] = {}
        self.default_key = model_registry.DEFAULT_KEY

        if mock:
            from realtime.mock import MockScorer
            self.model = MockScorer()
            logger.info("Engine: Using mock scorer")
        else:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required when mock=False")
            # A --ckpt that matches a known head is registered under that name,
            # so the dashboard's model tabs line up with what is loaded.
            key = model_registry.key_for_path(checkpoint_path) or "custom"
            if key == "custom":
                model_registry.REGISTRY["custom"] = (
                    "Custom", checkpoint_path, "Checkpoint passed with --ckpt.")
            self.default_key = key
            self.ensure_model(key, ckpt_override=checkpoint_path)
            self.model = self.heads[key]                     # legacy attribute
            self.config = self.head_configs[key]
            logger.info(f"Engine: default model '{key}' loaded from {checkpoint_path}")

        self.total_windows_scored = 0
        self.total_batches = 0

        # The front-end is ~300M parameters and takes tens of seconds to load.
        # Doing that lazily inside the first scoring batch made the first upload
        # look like a hang. preload() does it once, in a thread, at startup.
        self.warming = False
        self.warm = self.mock

    # ---- model management -------------------------------------------------

    def ensure_model(self, key: str, ckpt_override: str = None):
        """Load head `key` if it is not already resident. Returns the module.

        Raises with a readable message rather than returning None -- a silently
        absent head would score every window with whatever was loaded before,
        which is the worst possible failure here (a real number under the wrong
        model's name)."""
        if self.mock:
            return self.model
        if key in self.heads:
            return self.heads[key]

        entry = model_registry.REGISTRY.get(key)
        if entry is None and ckpt_override is None:
            raise KeyError(f"unknown model '{key}'")
        path = ckpt_override or entry[1]
        resolved = model_registry.resolve_ckpt(path)
        if not Path(resolved).exists():
            msg = f"checkpoint not found: {path}"
            self.load_errors[key] = msg
            raise FileNotFoundError(msg)

        from realtime.checkpoint import load_checkpoint
        try:
            head, cfg = load_checkpoint(resolved, self.device)
        except Exception as exc:
            self.load_errors[key] = str(exc)
            raise
        self.heads[key] = head
        self.head_configs[key] = cfg
        self.load_errors.pop(key, None)
        logger.info(f"Engine: loaded head '{key}' from {resolved}")
        return head

    async def preload(self):
        """Load the front-end and every head that exists, off the event loop.

        Called once at server start so the model cost is paid before anyone
        uploads anything, and so the dashboard can honestly say "warming up"
        instead of appearing frozen.
        """
        if self.mock or self.warm or self.warming:
            return
        self.warming = True
        try:
            for item in model_registry.catalogue():
                if not item["exists"]:
                    continue
                try:
                    await asyncio.to_thread(self.ensure_model, item["key"])
                except Exception as exc:
                    logger.warning(f"Engine: could not preload '{item['key']}': {exc}")

            from realtime import frontend
            logger.info("Engine: loading wav2vec2 front-end (this takes a moment)...")
            await asyncio.to_thread(frontend.load)
            # One throwaway pass so the first REAL window is not also paying
            # cuDNN autotuning and kernel compilation.
            await asyncio.to_thread(frontend.embed,
                                    [np.zeros(64000, dtype=np.float32)])
            self.warm = True
            logger.info("Engine: front-end warm. Uploads will score immediately.")
        except Exception as exc:
            logger.error(f"Engine: warm-up failed: {exc}", exc_info=True)
        finally:
            self.warming = False

    def model_catalogue(self) -> List[dict]:
        """What the dashboard shows in its model tabs."""
        out = []
        for item in model_registry.catalogue():
            item = dict(item)
            item["loaded"] = item["key"] in self.heads
            item["error"] = self.load_errors.get(item["key"])
            item["is_default"] = item["key"] == self.default_key
            out.append(item)
        return out

    def session_model_key(self, session) -> str:
        return getattr(session, "model_key", None) or self.default_key

    # ---- batching ---------------------------------------------------------

    async def _collect_batch(self):
        """Collect raw windows from all active sessions.

        Returns (call_ids, window_indices, model_keys, windows) or None.
        Embedding happens after collection so the whole batch goes through the
        front-end in ONE forward pass instead of one pass per window.
        """
        batch = []

        for call_id, session in self.sessions.items():
            from realtime.session import CallState
            if session.state not in [CallState.LISTENING, CallState.SCORING]:
                continue

            windows = await session.get_pending_windows()
            if not windows:
                continue

            key = self.session_model_key(session)
            taken = 0
            for window_idx, window in enumerate(windows):
                if len(batch) >= self.max_batch_size:
                    break
                batch.append((call_id,
                              session.window_count - len(windows) + window_idx,
                              key, window))
                taken += 1

            # Anything we could not fit goes back on the queue rather than being
            # dropped. get_pending_windows() already cleared it, so without this
            # every window past max_batch_size was lost.
            if taken < len(windows):
                await session.requeue_windows(windows[taken:])

            if len(batch) >= self.max_batch_size:
                break

        if not batch:
            return None

        call_ids, window_indices, model_keys, windows = zip(*batch)
        return list(call_ids), list(window_indices), list(model_keys), list(windows)

    async def _embed_windows(self, windows: List[np.ndarray]) -> np.ndarray:
        """4-second audio windows -> 1024-dim embeddings.

        In mock mode this stays random. In real mode it runs the SAME frozen
        wav2vec2 front-end our reported numbers came from -- returning random
        vectors here would produce confident, meaningless risk bands.

        Runs in a worker thread. This is not a micro-optimisation: a 300M-param
        forward pass on the event loop blocks EVERY http request for its whole
        duration, so the dashboard's own polling times out while the thing it
        is polling for is being computed. torch releases the GIL during compute,
        so the loop really does stay responsive.
        """
        if self.mock:
            return np.random.randn(len(windows), 1024).astype(np.float32)
        from realtime import frontend
        return await asyncio.to_thread(frontend.embed, windows)

    async def _embed_window(self, window: np.ndarray) -> np.ndarray:
        """Single-window embedding. Kept for callers/tests that still use it."""
        return (await self._embed_windows([window]))[0]

    async def _score_windows(self, embeddings: np.ndarray,
                             model_keys: List[str] = None) -> np.ndarray:
        """Score a batch, splitting it by which head each window asked for."""
        if self.mock:
            return np.asarray(self.model.score(embeddings), dtype=np.float32)

        if model_keys is None:
            model_keys = [self.default_key] * len(embeddings)

        scores = np.zeros(len(embeddings), dtype=np.float32)
        for key in sorted(set(model_keys)):
            rows = [i for i, k in enumerate(model_keys) if k == key]
            try:
                head = await asyncio.to_thread(self.ensure_model, key)
            except Exception as exc:
                logger.error(f"Engine: head '{key}' unavailable ({exc}); "
                             f"falling back to '{self.default_key}'")
                head = await asyncio.to_thread(self.ensure_model, self.default_key)

            scores[rows] = await asyncio.to_thread(self._head_forward, head,
                                                   embeddings[rows])

        return scores.astype(np.float32)

    def _head_forward(self, head, embeddings: np.ndarray) -> np.ndarray:
        """Synchronous head inference. Called via to_thread, never inline."""
        dev = next(head.parameters()).device if hasattr(head, "parameters") else (self.device or "cpu")
        with torch.no_grad():
            xb = torch.from_numpy(np.ascontiguousarray(embeddings)).float().to(dev)
            logits = head(xb)
            if hasattr(logits, "shape") and logits.shape[-1] == 2:
                out = torch.softmax(logits, dim=-1)[:, 1]
            else:
                out = torch.sigmoid(logits).squeeze(-1)
            return out.cpu().numpy().reshape(-1).astype(np.float32)

    # ---- session registry -------------------------------------------------

    async def add_session(self, session):
        """Register a new call for scoring."""
        self.sessions[session.call_id] = session
        logger.info(f"Engine: Added session {session.call_id} "
                    f"(model={self.session_model_key(session)})")

    async def remove_session(self, call_id: str):
        """Unregister a call."""
        if call_id in self.sessions:
            del self.sessions[call_id]
            logger.info(f"Engine: Removed session {call_id}")

    # ---- main loop --------------------------------------------------------

    async def run(self):
        """Main loop: collect → embed → score → broadcast."""
        logger.info("Engine: Starting scoring loop")

        try:
            while True:
                result = await self._collect_batch()

                if result is None:
                    await asyncio.sleep(self.batch_interval)
                    continue

                call_ids, window_indices, model_keys, windows = result
                embeddings = await self._embed_windows(windows)
                scores = await self._score_windows(embeddings, model_keys)

                for i, call_id in enumerate(call_ids):
                    session = self.sessions.get(call_id)
                    if session:
                        await session.record_score(window_indices[i], float(scores[i]))

                if self.on_broadcast:
                    # Keyed by call_id for backwards compatibility - the top-level
                    # window_idx/score are the LATEST window for that call - with
                    # every window of the batch preserved under "batch", so a
                    # multi-window batch no longer loses all but one score.
                    broadcast_data = {}
                    for i, call_id in enumerate(call_ids):
                        entry = broadcast_data.setdefault(
                            call_id, {"batch": [], "model": model_keys[i]})
                        session = self.sessions.get(call_id)
                        item = {
                            "window_idx": window_indices[i],
                            "score": float(scores[i]),
                            # Audio clock, not arrival clock - see
                            # Session.window_time. Without it a live chart drifts
                            # right by however far scoring is behind.
                            "t": session.window_time(window_indices[i]) if session else None,
                        }
                        entry["batch"].append(item)
                        entry["window_idx"] = item["window_idx"]
                        entry["score"] = item["score"]
                    await self.on_broadcast(broadcast_data)

                self.total_windows_scored += len(call_ids)
                self.total_batches += 1

                if self.total_batches % 100 == 0:
                    logger.info(f"Engine: {self.total_batches} batches, {len(self.sessions)} calls")

                # Work is queued: yield to the loop (so http stays responsive)
                # but do NOT sit out a full batch_interval. That fixed delay was
                # pure dead time -- a 66s clip is 125 windows, and at 8 windows
                # per half-second nap it spent 8 seconds asleep for no reason.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("Engine: Scoring loop cancelled")

    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            "total_batches": self.total_batches,
            "total_windows_scored": self.total_windows_scored,
            "active_calls": len(self.sessions),
            "loaded_models": sorted(self.heads.keys()),
            "warm": bool(self.warm),
            "warming": bool(self.warming),
            "default_model": self.default_key,
            "avg_windows_per_batch": (
                self.total_windows_scored / self.total_batches if self.total_batches > 0 else 0
            )
        }
