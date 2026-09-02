"""Bridge from the live path to the real detector in demo/score_file.py.

The frozen wav2vec2 front-end and the trained MLP head already live there and
are what the offline pipeline and the Streamlit demo use. The live path loads
THAT SAME module rather than reimplementing inference, so a checkpoint which
scores a clip offline scores it identically on a call. Nothing here duplicates
Yugal's model code - it only adapts its interface.

demo/ is not a package, so the module is loaded by file path.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCORE_FILE = Path(__file__).resolve().parent.parent / "demo" / "score_file.py"
_MODULE_KEY = "sonix_score_file"


def _module():
    """Import demo/score_file.py exactly once, keyed under its own name so it
    cannot collide with src/score_file.py (both exist - see FEATURES.txt 6h)."""
    mod = sys.modules.get(_MODULE_KEY)
    if mod is not None:
        return mod

    if not _SCORE_FILE.is_file():
        raise FileNotFoundError(f"detector module not found: {_SCORE_FILE}")

    spec = importlib.util.spec_from_file_location(_MODULE_KEY, _SCORE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {_SCORE_FILE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_KEY] = mod
    spec.loader.exec_module(mod)
    return mod


class WindowScorer:
    """Adapts score_file._score_windows to the engine's scorer interface.

    score(windows) -> np.ndarray of P(synthetic), one per window, same shape
    contract as MockScorer so the engine does not branch on which is loaded.
    Input is a stack of 64000-sample (4 s @ 16 kHz) float32 windows.
    """

    def __init__(self, module, ckpt_path: str):
        self._mod = module
        self._ckpt = ckpt_path

    def score(self, windows) -> np.ndarray:
        scores = self._mod._score_windows(list(windows), self._ckpt)
        return np.asarray(scores, dtype=np.float32)


def load_checkpoint(path, device=None):
    """Resolve and load the head so a missing/bad checkpoint fails at startup
    rather than silently mid-call.

    Returns (scorer, config). Raises FileNotFoundError if the checkpoint is not
    found - the server catches this and stays in mock mode with scoring
    disabled, which is what keeps a fake number off the screen.
    """
    mod = _module()
    if device:
        mod.configure(ckpt_path=path, device=device)

    resolved = mod._load_head(path)          # loads front-end + head, or raises

    # Read the raw checkpoint for flags score_file does not expose. A dev head
    # written by realtime/make_dev_head.py carries synthetic=True; everything
    # downstream keys off this so untrained scores can never pass as a verdict.
    import torch
    raw = torch.load(resolved, map_location="cpu", weights_only=False)

    config = {
        "ckpt": resolved,
        "synthetic": bool(raw.get("synthetic", False)),
        "dev_eer": mod.dev_eer(path),
        "window_samples": mod.WIN,
        "hop_samples": mod.HOP,
        "sample_rate": mod.TARGET_SR,
    }
    return WindowScorer(mod, resolved), config


def checkpoint_available(path=None) -> bool:
    """Cheap existence check - does not load torch or the front-end."""
    try:
        return _module().checkpoint_available(path)
    except Exception:
        return False


def discover_checkpoints() -> list[dict]:
    """Find available checkpoint heads without loading torch."""
    mod = _module()
    if hasattr(mod, "discover_checkpoints"):
        return mod.discover_checkpoints()
    return []
