"""
Per-call state machine for SONIX live detection.

Manages call lifecycle:
  CONNECTING → CONSENT_PENDING → LISTENING → SCORING → ENDED

Author: Claude Code
Date: 2026-08-31
"""

import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CallState(Enum):
    """Call lifecycle states."""
    CONNECTING = "connecting"           # Just created, setup phase
    CONSENT_PENDING = "consent_pending" # Pairing code sent, awaiting approval
    LISTENING = "listening"             # Accept audio, fill ringbuffer
    SCORING = "scoring"                 # Actively embed + score windows
    ENDED = "ended"                     # Call finished, archived
    ERROR = "error"                     # Unexpected error, call rejected


@dataclass
class AuditEntry:
    """Immutable log entry for consent/call events."""
    action: str                         # "pairing_sent", "pairing_approved", "window_scored", "call_ended"
    timestamp: str                      # ISO format
    metadata: Dict = field(default_factory=dict)  # Action-specific data


@dataclass
class CallMetadata:
    """Call metadata + consent audit trail."""
    call_id: str
    caller: str                         # Phone number or user ID
    pairing_code: str                   # 6-digit code
    pairing_expires_at: datetime
    started_at: datetime
    pairing_approved_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    audit: List[AuditEntry] = field(default_factory=list)


class Session:
    """
    Per-call session manager.

    Owns:
    - Call state machine (CONNECTING → ENDED)
    - Pairing code + consent audit
    - RingBuffer (4-second window collection)
    - VAD filter (silence detection)
    - Scored windows + metadata

    Does NOT own:
    - Model (that's engine.py)
    - Audio source (that's server.py)
    """

    def __init__(
        self,
        call_id: str,
        source,  # SourceAdapter (AudioSocket, WebRTC, file)
        pairing_code: str,
        pairing_expiry_sec: int = 120,
        vad_energy: float = None,
        model_key: str = None,
    ):
        """
        Args:
            call_id: Unique call identifier
            source: SourceAdapter instance (has .caller, .read())
            pairing_code: 6-digit approval code
            pairing_expiry_sec: Seconds until pairing code expires
            model_key: which trained head the engine should score this call
                with ("baseline"/"augmented"/"robust"). None = server default.
        """
        self.call_id = call_id
        self.source = source
        self.state = CallState.CONNECTING
        self.model_key = model_key

        # Set by the upload path so the dashboard can draw a real progress bar
        # instead of guessing when a file has finished streaming.
        self.expected_windows = None
        self.feed_done = False

        now = datetime.now()
        self.metadata = CallMetadata(
            call_id=call_id,
            caller=getattr(source, 'caller', 'unknown'),
            pairing_code=pairing_code,
            pairing_expires_at=now + timedelta(seconds=pairing_expiry_sec),
            started_at=now
        )

        # Import here to avoid circular imports
        from realtime.ringbuffer import RingBuffer
        from realtime.vad import VAD

        self.ringbuffer = RingBuffer()       # [capacity=64000 samples @ 16kHz = 4s]
        # Browser mic audio is often far quieter than the studio speech the
        # default floor assumes, so the gate is tunable from the server.
        self.vad = (VAD(threshold_energy=vad_energy) if vad_energy is not None
                    else VAD())

        # Windows that passed VAD (ready for scoring)
        self.pending_windows: List[np.ndarray] = []

        # Results: {window_idx: {"timestamp": float, "score": float}}
        self.scores: Dict[int, Dict] = {}

        # For ordering
        self.window_count = 0

        # Per-window telemetry for the dashboard: every window the ring buffer
        # emitted, whether or not VAD let it through. This is what makes the
        # silence gate visible instead of something we just claim happens.
        self.window_log: List[Dict] = []
        self.max_window_log = 600            # ~5 min at a 0.5s hop

        # Log the session creation
        self._add_audit("session_created", {"call_id": call_id})

    def _add_audit(self, action: str, metadata: dict = None):
        """Append immutable audit entry."""
        entry = AuditEntry(
            action=action,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.metadata.audit.append(entry)
        logger.info(f"[{self.call_id}] AUDIT: {action} | {metadata}")

    def get_pairing_status(self) -> dict:
        """Return pairing code + expiry for user approval."""
        if datetime.now() > self.metadata.pairing_expires_at:
            self.state = CallState.ERROR
            self._add_audit("pairing_expired")
            return None

        expires_in = (self.metadata.pairing_expires_at - datetime.now()).total_seconds()
        return {
            "call_id": self.call_id,
            "pairing_code": self.metadata.pairing_code,
            "expires_in": int(expires_in)
        }

    async def on_pairing_approved(self):
        """Called when user approves pairing code."""
        if self.state != CallState.CONSENT_PENDING:
            logger.warning(f"[{self.call_id}] Pairing approval in wrong state: {self.state}")
            return

        self.metadata.pairing_approved_at = datetime.now()
        self.state = CallState.LISTENING
        self._add_audit("pairing_approved", {"method": "mobile_scan"})
        logger.info(f"[{self.call_id}] Pairing approved → LISTENING")

    async def on_pairing_rejected(self):
        """Called when user rejects pairing code."""
        self.state = CallState.ERROR
        self._add_audit("pairing_rejected")
        logger.warning(f"[{self.call_id}] Pairing rejected")

    async def request_consent(self) -> bool:
        """
        Transition to CONSENT_PENDING, wait for pairing approval.

        Returns:
            True if approved within timeout, False if rejected/expired
        """
        if self.state != CallState.CONNECTING:
            logger.error(f"[{self.call_id}] Request consent in wrong state: {self.state}")
            return False

        self.state = CallState.CONSENT_PENDING
        self._add_audit("pairing_request", {"pairing_code": self.metadata.pairing_code})

        logger.info(f"[{self.call_id}] Waiting for pairing approval...")
        return True

    async def push_audio(self, samples: np.ndarray):
        """
        Receive audio chunk, push to ringbuffer.
        Only accepts audio if in LISTENING or SCORING state.

        Args:
            samples: [N] float32 array @ 16kHz
        """
        # Block audio if consent not given
        if self.state not in [CallState.LISTENING, CallState.SCORING]:
            logger.debug(f"[{self.call_id}] Ignoring audio in state {self.state}")
            return

        # Push to ringbuffer, get emitted windows
        self.ringbuffer.push(samples)
        emitted = self.ringbuffer.get_emitted_windows()

        # Filter by VAD, logging every window either way
        for window in emitted:
            passed = self.vad.is_speech(window)
            stats = dict(getattr(self.vad, "last_stats", {}))

            self.window_log.append({
                "t": datetime.now().timestamp(),
                "rms": stats.get("rms", 0.0),
                "peak": stats.get("peak", 0.0),
                "speech_ratio": stats.get("speech_ratio", 0.0),
                "vad_passed": bool(passed),
                "window_idx": self.window_count if passed else None,
            })
            if len(self.window_log) > self.max_window_log:
                del self.window_log[:-self.max_window_log]

            if passed:
                self.pending_windows.append(window)
                logger.debug(f"[{self.call_id}] Window {self.window_count} passed VAD")
                self.window_count += 1
            else:
                logger.debug(f"[{self.call_id}] Window rejected by VAD (silence)")

    async def get_pending_windows(self) -> List[np.ndarray]:
        """Return all windows pending scoring (passed VAD). Clears the list after return."""
        windows = self.pending_windows.copy()
        self.pending_windows.clear()
        return windows

    async def requeue_windows(self, windows: List[np.ndarray]):
        """
        Put windows back that the engine collected but could not fit in a batch.

        get_pending_windows() clears the queue, so without this any window past
        the engine's max_batch_size was silently dropped on the floor - audio
        that passed the silence gate and then vanished, with nothing logged.
        """
        if windows:
            self.pending_windows[:0] = windows

    async def record_score(self, window_idx: int, score: float, timestamp: float = None):
        """Record a scored window. Transitions to SCORING state if not already there."""
        if self.state == CallState.LISTENING:
            self.state = CallState.SCORING
            self._add_audit("first_score_recorded")

        self.scores[window_idx] = {
            "timestamp": timestamp or datetime.now().timestamp(),
            "score": float(score)
        }

        logger.debug(f"[{self.call_id}] Window {window_idx} scored: {score:.2%}")

    async def end_call(self):
        """Finalize call, archive metadata."""
        if self.state == CallState.ENDED:
            logger.warning(f"[{self.call_id}] Already ended")
            return

        self.state = CallState.ENDED
        self.metadata.ended_at = datetime.now()

        # Summary stats
        scores_list = [s["score"] for s in self.scores.values()]
        summary = {
            "duration_sec": (self.metadata.ended_at - self.metadata.started_at).total_seconds(),
            "windows_scored": len(self.scores),
            "mean_score": float(np.mean(scores_list)) if scores_list else None,
            "max_score": float(np.max(scores_list)) if scores_list else None,
            "min_score": float(np.min(scores_list)) if scores_list else None
        }

        self._add_audit("call_ended", summary)
        logger.info(f"[{self.call_id}] Call ended | {summary}")

    def to_dict(self) -> dict:
        """Serialize for WebSocket broadcast."""
        duration = (
            (self.metadata.ended_at or datetime.now()) - self.metadata.started_at
        ).total_seconds()

        scores_list = [s["score"] for s in self.scores.values()]

        return {
            "call_id": self.call_id,
            "caller": self.metadata.caller,
            "state": self.state.value,
            "duration": duration,
            "windows_scored": len(self.scores),
            "mean_score": float(np.mean(scores_list)) if scores_list else None,
            "max_score": float(np.max(scores_list)) if scores_list else None,
            "pairing_code": self.metadata.pairing_code if self.state == CallState.CONSENT_PENDING else None,
            "pairing_expires_in": int((self.metadata.pairing_expires_at - datetime.now()).total_seconds())
                if self.state == CallState.CONSENT_PENDING else None,
            "scores": self.scores
        }

    def telemetry(self, limit: int = 240) -> dict:
        """Everything the live dashboard needs for one call, in one payload."""
        rb = self.ringbuffer.stats() if hasattr(self.ringbuffer, "stats") else {}
        vd = self.vad.stats() if hasattr(self.vad, "stats") else {}
        return {
            "call_id": self.call_id,
            "caller": self.metadata.caller,
            "state": self.state.value,
            "model": self.model_key,
            "expected_windows": self.expected_windows,
            "feed_done": bool(self.feed_done),
            "pairing_code": self.metadata.pairing_code,
            "pairing_expires_in": max(
                0, int((self.metadata.pairing_expires_at - datetime.now()).total_seconds())
            ),
            "duration": (
                (self.metadata.ended_at or datetime.now()) - self.metadata.started_at
            ).total_seconds(),
            "ringbuffer": rb,
            "vad": vd,
            "windows": self.window_log[-limit:],
            "scores": self.scores,
        }

    def save_audit(self, output_dir: str = "outputs/calls"):
        """Persist call record to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        record = {
            "metadata": {
                "call_id": self.metadata.call_id,
                "caller": self.metadata.caller,
                "started_at": self.metadata.started_at.isoformat(),
                "ended_at": self.metadata.ended_at.isoformat() if self.metadata.ended_at else None,
                "pairing_approved_at": self.metadata.pairing_approved_at.isoformat()
                    if self.metadata.pairing_approved_at else None
            },
            "audit": [
                {"action": e.action, "timestamp": e.timestamp, "metadata": e.metadata}
                for e in self.metadata.audit
            ],
            "scores": self.scores
        }

        path = Path(output_dir) / f"{self.metadata.call_id}.json"
        with open(path, 'w') as f:
            json.dump(record, f, indent=2)

        logger.info(f"[{self.call_id}] Audit saved to {path}")
