#!/usr/bin/env python3
"""
score_file.py  --  SONIX / SIH26104   (the interface handed to Suryansh, T10)

The whole pipeline behind one function. This is all the demo/UI needs:

    from score_file import score_file
    scores = score_file("some_call.wav")   # -> [0.03, 0.05, 0.71, 0.88, ...]

    Returns one score per 4-second window at 0.5s hop (so ~2 scores/second).
    Higher = more likely FAKE (AI-cloned). Range 0..1 (calibrated probability).

Suryansh: until the real checkpoint exists you can keep mocking scores. The moment
outputs/models/head.pt is on the machine, `import score_file` and this returns real ones --
no other change to your Streamlit code. It runs on CPU (slow but fine for a demo);
if a GPU is present it uses it automatically.

Smoothing / hysteresis / risk-band mapping stays in the UI layer -- this returns the
raw per-window probabilities and nothing more, exactly as agreed.

CLI (for a quick check):
    python score_file.py some_call.wav
    python score_file.py some_call.wav --ckpt outputs/models/head.pt
"""

import sys
import numpy as np

TARGET_SR = 16000
WIN = 64000          # 4.0 s
HOP = 8000           # 0.5 s

_STATE = {"ckpt_path": "outputs/models/head.pt", "loaded": False}


def configure(ckpt_path="outputs/models/head.pt", model_name=None, device=None):
    """Optional: point at a different checkpoint before the first score_file()."""
    _STATE.update(ckpt_path=ckpt_path, loaded=False)
    if model_name:
        _STATE["model_name_override"] = model_name
    if device:
        _STATE["device_override"] = device


def _lazy_load():
    if _STATE["loaded"]:
        return
    import torch
    import torch.nn as nn
    from transformers import AutoFeatureExtractor, AutoModel

    ckpt = torch.load(_STATE["ckpt_path"], map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_name = _STATE.get("model_name_override") or ckpt.get(
        "front_end", "facebook/wav2vec2-xls-r-300m")
    device = _STATE.get("device_override") or (
        "cuda" if torch.cuda.is_available() else "cpu")

    fe = AutoFeatureExtractor.from_pretrained(model_name)
    frontend = AutoModel.from_pretrained(model_name).eval().to(device)
    for p in frontend.parameters():
        p.requires_grad_(False)

    head = nn.Sequential(
        nn.Linear(cfg["in_dim"], cfg["hidden"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden"], 1),
    ).to(device).eval()
    head.load_state_dict(ckpt["state_dict"])

    _STATE.update(
        loaded=True, torch=torch, fe=fe, frontend=frontend, head=head,
        device=device,
        mu=np.asarray(ckpt["mu"], np.float32),
        sd=np.asarray(ckpt["sd"], np.float32),
    )


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
    """Yield fixed 64000-sample windows at 0.5 s hop. Short audio -> one padded."""
    if len(wav) < WIN:
        yield np.pad(wav, (0, WIN - len(wav)))
        return
    start = 0
    while start + WIN <= len(wav):
        yield wav[start:start + WIN]
        start += HOP


def score_file(wav_path, batch=8) -> list:
    """One calibrated fake-probability per 4 s window (0.5 s hop). Higher = faker."""
    _lazy_load()
    torch = _STATE["torch"]
    fe, frontend, head, device = (_STATE["fe"], _STATE["frontend"],
                                  _STATE["head"], _STATE["device"])
    mu, sd = _STATE["mu"], _STATE["sd"]

    wav = _load_audio(wav_path)
    wins = list(_windows(wav))
    scores = []
    for i in range(0, len(wins), batch):
        mb = wins[i:i + batch]
        inputs = fe(mb, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        iv = inputs["input_values"].to(device)
        with torch.no_grad():
            hidden = frontend(iv).last_hidden_state          # (b, T, 1024)
            pooled = hidden.mean(dim=1).cpu().numpy().astype(np.float32)
            pooled = (pooled - mu) / sd
            logits = head(torch.from_numpy(pooled).float().to(device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        scores.extend(float(p) for p in probs)
    return scores


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default="outputs/models/head.pt")
    args = ap.parse_args()
    configure(ckpt_path=args.ckpt)
    s = score_file(args.wav)
    print(f"{len(s)} windows (~{len(s) * 0.5:.1f}s of coverage)")
    print("scores:", [round(x, 3) for x in s])
    if s:
        print(f"mean={np.mean(s):.3f}  max={np.max(s):.3f}  "
              f"final-window={s[-1]:.3f}")
