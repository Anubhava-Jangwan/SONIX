"""SONIX client SDK — how a bank, contact centre or telco integrates voice-clone detection.

SONIX is a security LAYER, not an application. The dashboard in realtime/live_ui.py is one
client of this API; a production integration is the code below, called from inside whatever
system already handles the call.

    pip install requests
    from sonix_sdk import SonixClient

    sonix = SonixClient("http://sonix.internal:8000")

    # --- post-call / recorded ---------------------------------------------
    result = sonix.score_file("call_88213.wav")
    if result.band == "RED":
        hold_transaction(reason=f"voice-clone risk {result.mean:.0%}")

    # --- live, during the call --------------------------------------------
    with sonix.live_call(caller="+91XXXXXXXXXX") as call:
        for chunk in telephony_stream:          # 16 kHz mono PCM
            call.push(chunk)
            if call.band == "RED":
                warn_agent("Caller may be using a cloned voice")
                break

Every call is consent-gated and audited server-side. No audio is retained past the call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Operator-configurable. These are the data-driven defaults from our eval
# distribution; a bank tunes them per workflow (stricter for fund transfers
# than for general enquiries).
AMBER, RED = 0.10, 0.90


def band_for(score, amber=AMBER, red=RED):
    if score is None:
        return "UNKNOWN"
    return "RED" if score >= red else ("AMBER" if score >= amber else "GREEN")


@dataclass
class ScoreResult:
    """One scored call."""
    call_id: str
    model: str
    scores: list = field(default_factory=list)   # per 4-second window
    duration_s: float = 0.0

    @property
    def mean(self):
        return sum(self.scores) / len(self.scores) if self.scores else None

    @property
    def peak(self):
        return max(self.scores) if self.scores else None

    @property
    def band(self):
        """GREEN / AMBER / RED, from the smoothed score."""
        return band_for(self.smoothed)

    @property
    def smoothed(self):
        """Mean of the last 5 windows — the same smoothing the live engine uses,
        so a single odd window cannot flip a verdict."""
        if not self.scores:
            return None
        tail = self.scores[-5:]
        return sum(tail) / len(tail)

    def as_dict(self):
        return {"call_id": self.call_id, "model": self.model, "band": self.band,
                "mean": self.mean, "peak": self.peak,
                "windows": len(self.scores), "duration_s": self.duration_s}


class LiveCall:
    """A call being scored while it is still in progress."""

    def __init__(self, client, call_id, expected_windows=0):
        self._c = client
        self.call_id = call_id
        self.expected_windows = expected_windows
        self.scores = []

    def refresh(self):
        """Pull the scores recorded so far. Cheap; poll a few times a second."""
        r = self._c._get("/api/telemetry", params={"call_id": self.call_id, "limit": 5000})
        call = (r.get("calls") or {}).get(self.call_id)
        if call:
            self.scores = [float(v["score"]) for _, v in
                           sorted((call.get("scores") or {}).items(), key=lambda kv: int(kv[0]))]
        return self.scores

    @property
    def band(self):
        self.refresh()
        if not self.scores:
            return "UNKNOWN"
        tail = self.scores[-5:]
        return band_for(sum(tail) / len(tail))

    @property
    def risk(self):
        """Current smoothed probability the caller's voice is synthetic."""
        self.refresh()
        if not self.scores:
            return None
        tail = self.scores[-5:]
        return sum(tail) / len(tail)

    def end(self):
        try:
            self._c._post("/api/end-call", {"call_id": self.call_id})
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.end()
        return False


class SonixClient:
    """Talks to a SONIX scoring service.

    Deployment note: this service is self-hosted. It runs as a container inside
    the bank's or operator's own infrastructure, so call audio never leaves
    their network.
    """

    def __init__(self, base_url="http://localhost:8000", timeout=30, model=None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model                       # None = server default

    # ---- plumbing --------------------------------------------------------
    def _get(self, path, params=None):
        r = requests.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path, payload):
        r = requests.post(f"{self.base}{path}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- service ---------------------------------------------------------
    def health(self):
        """Service status: is a trained model loaded and warm?"""
        return self._get("/api/status")

    def models(self):
        """Detector models this service can score with."""
        return self._get("/api/models")

    def ready(self):
        m = self.models()
        return bool(m.get("warm")) and not m.get("mock", True)

    # ---- scoring ---------------------------------------------------------
    def score_file(self, path, model=None, wait=True, poll=0.4, timeout=600):
        """Score a recording. Returns a ScoreResult.

        Use for post-call review, dispute investigation, or batch audit of
        recorded lines.
        """
        with open(path, "rb") as fh:
            r = requests.post(
                f"{self.base}/api/score-file",
                files={"file": (Path(path).name, fh.read())},
                data={"model": model or self.model or ""},
                timeout=max(self.timeout, 120))
        r.raise_for_status()
        started = r.json()

        res = ScoreResult(call_id=started["call_id"],
                          model=started.get("model") or "default",
                          duration_s=float(started.get("duration_s") or 0))
        if not wait:
            return res

        expected = max(1, int(started.get("expected_windows") or 1))
        deadline, stall, last = time.time() + timeout, time.time(), 0
        while time.time() < deadline:
            call = LiveCall(self, res.call_id, expected)
            scores = call.refresh()
            if len(scores) > last:
                last, stall = len(scores), time.time()
            if len(scores) >= expected or time.time() - stall > 15:
                res.scores = scores
                break
            time.sleep(poll)
        self._post("/api/end-call", {"call_id": res.call_id})
        return res

    def live_call(self, caller="unknown", model=None):
        """Open a call scored while it is still running.

        Audio is pushed over the WebSocket at /ws as 16 kHz mono PCM. In a
        telephony integration this is wired to the media stream; the returned
        object exposes .risk and .band at any moment during the call.
        """
        raise NotImplementedError(
            "Live streaming is wired through the WebSocket at /ws. See "
            "realtime/miccapture.py for a reference implementation, and "
            "docs/API.md for the message protocol.")


if __name__ == "__main__":
    import sys
    c = SonixClient(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
    print("service:", json.dumps(c.health(), indent=2)[:400])
    if len(sys.argv) > 2:
        r = c.score_file(sys.argv[2])
        print(json.dumps(r.as_dict(), indent=2))
