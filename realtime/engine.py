"""Real-time scoring engine: model owner + batch inference."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Single asyncio task that batches windows across concurrent calls."""

    def __init__(
        self,
        mock: bool = True,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
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

            for window_idx, window in enumerate(windows):
                embedding = await self._embed_window(window)
                batch.append((call_id, session.window_count - len(windows) + window_idx, embedding))

                if len(batch) >= self.max_batch_size:
                    break

            if len(batch) >= self.max_batch_size:
                break

        if not batch:
            return None

        call_ids, window_indices, embeddings = zip(*batch)
        embeddings_array = np.stack(embeddings, axis=0)
        return list(call_ids), list(window_indices), embeddings_array

    async def _embed_window(self, window: np.ndarray) -> np.ndarray:
        """Extract 1024-dim embedding from 4-second audio window."""
        return np.random.randn(1024).astype(np.float32)

    async def _score_windows(self, embeddings: np.ndarray) -> np.ndarray:
        """Score a batch of embeddings."""
        if self.mock:
            scores = self.model.score(embeddings)
        else:
            with torch.no_grad():
                embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
                logits = self.model(embeddings_tensor)

                if logits.shape[-1] == 2:
                    scores_tensor = torch.softmax(logits, dim=-1)[:, 1]
                else:
                    scores_tensor = torch.sigmoid(logits).squeeze(-1)

                scores = scores_tensor.cpu().numpy()

        return scores.astype(np.float32)

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
                    broadcast_data = {
                        call_id: {
                            "window_idx": window_indices[i],
                            "score": float(scores[i])
                        }
                        for i, call_id in enumerate(call_ids)
                    }
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
