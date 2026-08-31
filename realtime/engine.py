"""Real-time scoring engine: model owner + batch inference."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

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

        if mock:
            from realtime.mock import MockScorer
            self.model = MockScorer()
            logger.info("Engine: Using mock scorer")
        else:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path required when mock=False")
            from realtime.checkpoint import load_checkpoint
            self.model, self.config = load_checkpoint(checkpoint_path, device)
            logger.info(f"Engine: Loaded model from {checkpoint_path}")

        self.total_windows_scored = 0
        self.total_batches = 0

    async def add_session(self, session):
        """Register a new call for scoring."""
        self.sessions[session.call_id] = session
        logger.info(f"Engine: Added session {session.call_id}")

    async def remove_session(self, call_id: str):
        """Unregister a call."""
        if call_id in self.sessions:
            del self.sessions[call_id]
            logger.info(f"Engine: Removed session {call_id}")

    async def _collect_batch(self) -> Optional[Tuple[List[str], List[int], np.ndarray]]:
        """Collect windows from all active sessions."""
        batch = []

        for call_id, session in self.sessions.items():
            from realtime.session import CallState
            if session.state not in [CallState.LISTENING, CallState.SCORING]:
                continue

            windows = await session.get_pending_windows()
            if not windows:
                continue

            taken = 0
            for window_idx, window in enumerate(windows):
                if len(batch) >= self.max_batch_size:
                    break
                # Raw 64000-sample window goes straight into the batch. The
                # wav2vec2 front-end runs inside the scorer, batched, rather
                # than once per window here.
                batch.append((call_id, session.window_count - len(windows) + window_idx, window))
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

        call_ids, window_indices, embeddings = zip(*batch)
        embeddings_array = np.stack(embeddings, axis=0)
        return list(call_ids), list(window_indices), embeddings_array

    async def _score_windows(self, windows: np.ndarray) -> np.ndarray:
        """Score a batch of 4-second windows -> P(synthetic), one per window.

        Both MockScorer and WindowScorer expose .score(batch), so there is no
        branch here. The real path runs the frozen front-end and the trained
        head inside demo/score_file.py - the same code the offline pipeline
        uses, so a live score and an offline score of the same audio agree.

        This blocks the event loop for the length of one forward pass, so it
        runs in a worker thread; otherwise audio ingest stalls while the GPU
        works.
        """
        scores = await asyncio.to_thread(self.model.score, windows)
        return np.asarray(scores, dtype=np.float32)

    async def run(self):
        """Main loop: collect → batch → score → broadcast."""
        logger.info("Engine: Starting scoring loop")

        try:
            while True:
                result = await self._collect_batch()

                if result is None:
                    await asyncio.sleep(self.batch_interval)
                    continue

                call_ids, window_indices, embeddings = result
                scores = await self._score_windows(embeddings)

                for i, call_id in enumerate(call_ids):
                    session = self.sessions.get(call_id)
                    if session:
                        window_idx = window_indices[i]
                        score = float(scores[i])
                        await session.record_score(window_idx, score)

                if self.on_broadcast:
                    # Keyed by call_id for backwards compatibility - the top-level
                    # window_idx/score are the LATEST window for that call - with
                    # every window of the batch preserved under "batch", so a
                    # multi-window batch no longer loses all but one score.
                    broadcast_data = {}
                    for i, call_id in enumerate(call_ids):
                        entry = broadcast_data.setdefault(call_id, {"batch": []})
                        item = {"window_idx": window_indices[i], "score": float(scores[i])}
                        entry["batch"].append(item)
                        entry["window_idx"] = item["window_idx"]
                        entry["score"] = item["score"]
                    await self.on_broadcast(broadcast_data)

                self.total_windows_scored += len(call_ids)
                self.total_batches += 1

                if self.total_batches % 100 == 0:
                    logger.info(f"Engine: {self.total_batches} batches, {len(self.sessions)} calls")

                await asyncio.sleep(self.batch_interval)

        except asyncio.CancelledError:
            logger.info("Engine: Scoring loop cancelled")

    def get_stats(self) -> dict:
        """Return engine statistics."""
        return {
            "total_batches": self.total_batches,
            "total_windows_scored": self.total_windows_scored,
            "active_calls": len(self.sessions),
            "avg_windows_per_batch": (
                self.total_windows_scored / self.total_batches if self.total_batches > 0 else 0
            )
        }
