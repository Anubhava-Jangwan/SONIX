"""Adapter so Suryansh's demo UI can reach the real detector.

v9 LIVE path:     from model import score_stream   # generator, one score per window
Offline/bench:    from model import score_file     # full list at once

Running REAL mode needs, on THIS machine:
  * pip install torch transformers   (beyond the demo requirements)
  * the trained checkpoint at  outputs/models/head.pt
Flip real mode only on a machine that has those (your GPU laptop / demo laptop).
Mock mode in the UI needs none of this.
"""
from score_file import score_file, score_stream  # noqa: F401
