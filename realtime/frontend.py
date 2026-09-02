"""Frozen wav2vec2 front-end for the live engine.

Same computation as demo/score_file.py, which is the path our reported numbers
came from: mean-pool the last hidden state over time -> one 1024-dim vector per
4-second window. Loaded once and kept resident; it is ~300M parameters and must
not be reloaded per window.
"""

import numpy as np

TARGET_SR = 16000
_STATE = {"loaded": False}


def load(model_name="facebook/wav2vec2-xls-r-300m", device=None, half=True):
    """Load the frozen front-end once. Safe to call repeatedly."""
    if _STATE["loaded"]:
        return
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = bool(half) and str(device).startswith("cuda")

    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    if use_half:
        model.half()
    for p in model.parameters():
        p.requires_grad_(False)

    _STATE.update(loaded=True, torch=torch, fe=fe, model=model, device=device,
                  dtype=torch.float16 if use_half else torch.float32)


def embed(windows):
    """list of (64000,) float32 windows -> (n, 1024) float32 embeddings."""
    load()
    torch = _STATE["torch"]
    fe, model = _STATE["fe"], _STATE["model"]
    device, dtype = _STATE["device"], _STATE["dtype"]

    wins = [np.asarray(w, dtype=np.float32) for w in windows]
    inputs = fe(wins, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    iv = inputs["input_values"].to(device=device, dtype=dtype)
    kw = {}
    if "attention_mask" in inputs:
        kw["attention_mask"] = inputs["attention_mask"].to(device)
    with torch.no_grad():
        hidden = model(iv, **kw).last_hidden_state        # (b, T, 1024)
        pooled = hidden.mean(dim=1)
    return pooled.float().cpu().numpy().astype(np.float32)
