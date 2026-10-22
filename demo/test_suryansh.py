import numpy as np
from windowing import make_windows
from risk import process_scores
from streaming import demo_score_stream

# Windowing
x = np.zeros(30 * 16000, dtype=np.float32)
ws = list(make_windows(x, 16000, 4.0, 0.5))
assert len(ws) == 54
assert all(w.shape == (64000,) for _, w in ws)
print("PASS: 30-second audio -> 54 padded windows")

# Risk pipeline
raw = np.array([0.1, 0.2, 0.1, 0.8, 0.9, 0.85, 0.9])
smoothed, bands = process_scores(raw, 0.45, 0.70)
assert len(smoothed) == len(raw) and len(bands) == len(raw)
assert bands[:5] == ["GREEN"] * 5
print("PASS: smoothing + hysteresis + warm-up")

# True incremental mock stream: the consumer receives one score at a time.
received = []
for idx, score in demo_score_stream(5, "genuine_01.wav", step_delay_s=0):
    received.append((idx, score))
assert len(received) == 5
assert [i for i, _ in received] == list(range(5))
print("PASS: score stream yields one result at a time")

# Model registry: the demo must expose every head the live server knows about.
# demo/core.py used to keep its own 3-head dict, so the UI could not reach v3
# (the server default) or robust_v2 -- and nothing failed to say so.
import ui
import core
from core import available_models, default_index
from realtime.models import DEFAULT_KEY, REGISTRY

assert core.MODELS is REGISTRY, "demo/core.py is not reading realtime.models.REGISTRY"
print(f"PASS: demo and live server share one registry ({len(REGISTRY)} heads)")

keys = list(available_models())
assert keys, "no checkpoint on disk -- cannot check the picker"
if DEFAULT_KEY in keys:
    assert keys[default_index(keys)] == DEFAULT_KEY, "picker does not open on DEFAULT_KEY"
    print(f"PASS: picker opens on DEFAULT_KEY ({DEFAULT_KEY}), {len(keys)} heads on disk")

assert len(ui.SERIES) >= len(REGISTRY), (
    f"compare chart has {len(ui.SERIES)} colours for {len(REGISTRY)} heads -- "
    f"colours would cycle and two heads would draw identically")
print(f"PASS: {len(ui.SERIES)} series colours for {len(REGISTRY)} heads")
