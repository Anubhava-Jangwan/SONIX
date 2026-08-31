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
