#!/usr/bin/env python3
"""
score_file.py  --  SONIX / SIH26104   (the interface handed to Suryansh, T10)

The whole pipeline behind one function. This is all the demo/UI needs:

    from score_file import score_file
    scores = score_file("some_call.wav")                                 # baseline head
    scores = score_file("some_call.wav", ckpt_path="outputs/models/head_aug.pt")  # augmented head

    Returns one score per 4-second window at 0.5s hop (so ~2 scores/second).
    Higher = more likely FAKE (AI-cloned). Range 0..1 (calibrated probability).

TWO HEADS, ONE FRONT-END
    The frozen wav2vec2 XLS-R front-end (300M params) is loaded ONCE and shared.
    Each checkpoint -- baseline head.pt, augmented head_aug.pt -- only swaps in its
    own small MLP head plus its input standardiser (mu/sd). So the demo can score
    the same clip through both models back-to-back without reloading the front-end.

CHECKPOINT PATH RESOLUTION
    Paths are resolved robustly: if the given path is not found as-is we search
    upward from this file's directory and from the current working directory for
    outputs/models/<name>. So `streamlit run app.py` finds the checkpoint whether
    it is launched from demo/ or from the repo root (fixes the old
    FileNotFoundError: outputs/models/head.pt).

Smoothing / hysteresis / risk-band mapping stays in the UI layer -- this returns
the raw per-window probabilities and nothing more, exactly as agreed.

CLI (for a quick check):
    python score_file.py some_call.wav
    python score_file.py some_call.wav --ckpt outputs/models/head_aug.pt
"""

import sys
from pathlib import Path

import numpy as np

TARGET_SR = 16000
WIN = 64000          # 4.0 s
HOP = 8000           # 0.5 s

DEFAULT_CKPT = "outputs/models/head.pt"
MODEL_DIRS = ("models", "outputs/models")

# Shared frozen front-end (loaded once) + a cache of small heads keyed by checkpoint.
_FE = {"loaded": False}
_HEADS = {}                                  # resolved_ckpt(str) -> {"head","mu","sd","dev_eer"}
_STATE = {"default_ckpt": DEFAULT_CKPT}      # kept for backward compatibility

# ---------------------------------------------------------------------------
# Voice-activity (energy) gate.
# Near-silent windows are NOT sent to the model. Real recordings have quiet
# gaps, and the model has no calibrated behaviour there -- the feature
# extractor normalises every window, which amplifies noise in silence and
# pushes the score toward "fake". Skipping them removes a major source of
# false alarms on genuine audio (and makes scoring faster).
# ---------------------------------------------------------------------------
_VAD = {"enabled": True, "dbfs": -45.0, "silence_score": 0.0}


def set_vad(enabled=True, dbfs=-45.0, silence_score=0.0):
    """Turn the silence gate on/off and set its threshold in dBFS RMS."""
    _VAD.update(enabled=bool(enabled), dbfs=float(dbfs),
                silence_score=float(silence_score))


def get_vad():
    """Current gate settings (for the UI)."""
    return dict(_VAD)


def rms_dbfs(w):
    """RMS energy of one window in dBFS. 0 = full scale, -60 = very quiet."""
    a = np.asarray(w, dtype=np.float64)
    r = float(np.sqrt(np.mean(np.square(a)) + 1e-12))
    return 20.0 * np.log10(r + 1e-12)



def configure(ckpt_path=DEFAULT_CKPT, model_name=None, device=None):
    """Optional: set the DEFAULT checkpoint / front-end / device used when a caller
    does not pass ckpt_path explicitly. Safe to call more than once."""
    _STATE["default_ckpt"] = ckpt_path
    if model_name:
        _STATE["model_name_override"] = model_name
    if device:
        _STATE["device_override"] = device


def _candidate_roots():
    bases = [Path.cwd(), Path(__file__).resolve().parent]
    seen = set()
    for base in bases:
        for d in [base, *base.parents]:
            resolved = d.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield d


def resolve_ckpt(ckpt_path=None) -> str:
    """Return an existing checkpoint path. Tries the path as-is, then searches
    upward for models/<name> and outputs/models/<name>. Raises FileNotFoundError
    if nothing matches."""
    ckpt_path = ckpt_path or _STATE.get("default_ckpt", DEFAULT_CKPT)
    p = Path(ckpt_path)
    if p.is_file():
        return str(p.resolve())

    name = p.name
    tried = []
    for root in _candidate_roots():
        for model_dir in MODEL_DIRS:
            cand = root / model_dir / name
            tried.append(cand)
            if cand.is_file():
                return str(cand.resolve())
    # last resort: the raw relative path joined onto each base
    for root in _candidate_roots():
        cand = root / ckpt_path
        if cand.is_file():
            return str(cand.resolve())

    raise FileNotFoundError(
        f"checkpoint not found: {ckpt_path}  "
        f"(also searched upward for models/{name} and outputs/models/{name})"
    )


def checkpoint_available(ckpt_path=None) -> bool:
    """Cheap existence check for the UI -- does NOT load torch."""
    try:
        resolve_ckpt(ckpt_path)
        return True
    except FileNotFoundError:
        return False


def discover_checkpoints() -> list[dict]:
    """Find checkpoint heads in models/ and outputs/models/.

    Returns dictionaries with stable IDs for UI/API use. This is a cheap file
    scan; it does not import torch or load model weights.
    """
    found = {}
    for root in _candidate_roots():
        for model_dir in MODEL_DIRS:
            folder = root / model_dir
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.pt")):
                resolved = str(path.resolve())
                found.setdefault(resolved, {
                    "id": path.stem,
                    "label": label_for_checkpoint(path),
                    "path": resolved,
                    "filename": path.name,
                })
    return sorted(found.values(), key=lambda item: item["label"].lower())


def label_for_checkpoint(path) -> str:
    name = Path(path).stem
    known = {
        "head": "Baseline",
        "head_aug": "Augmented",
        "head_robust": "Robust",
    }
    if name in known:
        return known[name]
    return name.replace("head_", "").replace("_", " ").strip().title() or name


def _load_frontend():
    """Load the frozen wav2vec2 front-end exactly once and keep it resident."""
    if _FE["loaded"]:
        return
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    model_name = _STATE.get("model_name_override", "facebook/wav2vec2-xls-r-300m")
    device = _STATE.get("device_override") or (
        "cuda" if torch.cuda.is_available() else "cpu")

    fe = AutoFeatureExtractor.from_pretrained(model_name)
    frontend = AutoModel.from_pretrained(model_name).eval().to(device)
    for p in frontend.parameters():
        p.requires_grad_(False)

    _FE.update(loaded=True, torch=torch, fe=fe, frontend=frontend, device=device)


def _load_head(ckpt_path=None) -> str:
    """Load (and cache) the small MLP head + standardiser for one checkpoint.
    The shared front-end is loaded on first use. Returns the resolved ckpt path."""
    resolved = resolve_ckpt(ckpt_path)
    if resolved in _HEADS:
        return resolved

    _load_frontend()
    torch = _FE["torch"]
    import torch.nn as nn

    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    head = nn.Sequential(
        nn.Linear(cfg["in_dim"], cfg["hidden"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden"], 1),
    ).to(_FE["device"]).eval()
    head.load_state_dict(ckpt["state_dict"])

    _HEADS[resolved] = {
        "head": head,
        "mu": np.asarray(ckpt["mu"], np.float32),
        "sd": np.asarray(ckpt["sd"], np.float32),
        "dev_eer": ckpt.get("dev_eer"),
    }
    return resolved


def dev_eer(ckpt_path=None):
    """Return the checkpoint's stored dev EER (loads the head; None if absent)."""
    resolved = _load_head(ckpt_path)
    return _HEADS[resolved].get("dev_eer")


def _load_audio(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import torch
            import torchaudio.functional as AF
            wav = AF.resample(torch.from_numpy(wav).float().unsqueeze(0),
                              sr, TARGET_SR).squeeze(0).numpy()
        except Exception:
            n_out = int(round(len(wav) * TARGET_SR / sr))
            wav = np.interp(np.linspace(0, 1, n_out, endpoint=False),
                            np.linspace(0, 1, len(wav), endpoint=False),
                            wav).astype(np.float32)
    return wav


def _windows(wav):
    """Yield 64000-sample windows at 0.5 s hop, higher = more likely fake.

    PADDING MATTERS. Measured on a real 66 s human recording: full windows
    scored 0.12 (correctly REAL) while 1-second zero-padded excerpts of the SAME
    voice scored 0.88 (wrongly FAKE). Zero-padding leaves a window mostly
    silence, the feature extractor normalises across those zeros and amplifies
    what little speech is there, and out-of-domain audio gets pushed toward
    "fake". So:

      * clip >= 4 s : emit full windows only. The trailing partial window is
        dropped -- at a 0.5 s hop the previous window already covers that audio,
        so nothing is lost.
      * clip <  4 s : REPEAT-pad (loop the audio) instead of zero-padding, so
        the window is full of real speech rather than half silence.
    """
    n = len(wav)
    if n == 0:
        yield np.zeros(WIN, dtype=np.float32)
        return

    if n < WIN:                          # short clip: repeat-pad, never zero-pad
        reps = int(np.ceil(WIN / n))
        yield np.tile(wav, reps)[:WIN].astype(np.float32)
        return

    start = 0
    while start + WIN <= n:
        yield wav[start:start + WIN]
        start += HOP


def _score_windows(wins, ckpt_path=None) -> list:
    """Score a list of 64000-sample windows -> list[float] fake-probabilities,
    using the shared front-end and the head for `ckpt_path`.

    Windows whose energy falls below the VAD threshold are never sent to the
    model; they return `silence_score` (default 0.0 = green). Shared by
    score_file (batched, offline) and score_stream (one at a time, live)."""
    wins = list(wins)
    n = len(wins)
    if n == 0:
        return []

    if _VAD["enabled"]:
        keep = [i for i in range(n) if rms_dbfs(wins[i]) > _VAD["dbfs"]]
    else:
        keep = list(range(n))

    out = [float(_VAD["silence_score"])] * n
    if not keep:
        return out

    resolved = _load_head(ckpt_path)
    torch = _FE["torch"]
    fe, frontend, device = _FE["fe"], _FE["frontend"], _FE["device"]
    h = _HEADS[resolved]
    head, mu, sd = h["head"], h["mu"], h["sd"]

    sub = [wins[i] for i in keep]
    inputs = fe(sub, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    iv = inputs["input_values"].to(device)
    with torch.no_grad():
        hidden = frontend(iv).last_hidden_state              # (b, T, 1024)
        pooled = hidden.mean(dim=1).cpu().numpy().astype(np.float32)
        pooled = (pooled - mu) / sd
        logits = head(torch.from_numpy(pooled).float().to(device)).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    for i, p in zip(keep, probs):
        out[i] = float(p)
    return out


def score_file(wav_path, batch=8, ckpt_path=None) -> list:
    """OFFLINE interface: one calibrated fake-probability per 4 s window (0.5 s
    hop) as a full list. Higher = faker. `ckpt_path` picks the model (default =
    baseline head.pt); pass outputs/models/head_aug.pt for the augmented head."""
    _load_head(ckpt_path)
    wav = _load_audio(wav_path)
    wins = list(_windows(wav))
    scores = []
    for i in range(0, len(wins), batch):
        scores.extend(_score_windows(wins[i:i + batch], ckpt_path))
    return scores


def score_stream(wav_path, ckpt_path=None):
    """LIVE interface (Suryansh's v9 contract): a generator that yields ONE
    fake-probability per 4 s window, in order, computing each on the fly. The
    front-end + head are loaded once (via _load_head) and stay resident for the
    whole call. `ckpt_path` picks the model. Higher = more likely fake.

        for score in score_stream("call.wav", ckpt_path="outputs/models/head_aug.pt"):
            ...   # UI updates as each window's score arrives
    """
    _load_head(ckpt_path)
    wav = _load_audio(wav_path)
    for w in _windows(wav):
        yield _score_windows([w], ckpt_path)[0]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to use (default: baseline outputs/models/head.pt)")
    args = ap.parse_args()
    s = score_file(args.wav, ckpt_path=args.ckpt)
    print(f"{len(s)} windows (~{len(s) * 0.5:.1f}s of coverage)")
    print("scores:", [round(x, 3) for x in s])
    if s:
        print(f"mean={np.mean(s):.3f}  max={np.max(s):.3f}  "
              f"final-window={s[-1]:.3f}")
