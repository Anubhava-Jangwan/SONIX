"""Device pairing with 6-digit codes.

The code is what an operator reads off the capture page and approves in the
dashboard. It is a consent prompt, not authentication -- see docs/AUDIT_PHASE1.md
-- but it should still not be guessable or brute-forceable.

Previously this used `random.randint`, i.e. Mersenne Twister: not a CSPRNG, and
recoverable from a handful of observed outputs. There was also no attempt
counter and no issued-at time, so `expiry_sec` was stored and never enforced.
"""

import secrets
import time


class PairingCodeManager:
    def __init__(self, expiry_sec=120, max_attempts=5):
        self.expiry_sec = expiry_sec
        self.max_attempts = max_attempts
        # code -> {"issued_at": float, "attempts": int}
        self._issued = {}

    def generate(self):
        """A cryptographically random 6-digit code, recorded so it can expire."""
        self._prune()
        code = f"{secrets.randbelow(900000) + 100000}"
        self._issued[code] = {"issued_at": time.monotonic(), "attempts": 0}
        return code

    def check(self, code):
        """Verify a submitted code. Constant-time compare, counted, expiring.

        Returns True only for a live, unexpired code that has not exhausted its
        attempt budget. Wrong guesses burn the budget for the code they were
        aimed at, so 900k blind guesses are not available.
        """
        self._prune()
        for known, meta in self._issued.items():
            if meta["attempts"] >= self.max_attempts:
                continue
            if secrets.compare_digest(str(code), known):
                del self._issued[known]        # single use
                return True
        for meta in self._issued.values():
            meta["attempts"] += 1
        return False

    def revoke(self, code):
        self._issued.pop(str(code), None)

    def _prune(self):
        now = time.monotonic()
        for code, meta in list(self._issued.items()):
            if now - meta["issued_at"] > self.expiry_sec:
                del self._issued[code]
