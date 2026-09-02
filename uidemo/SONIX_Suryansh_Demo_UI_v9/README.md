# SONIX — Suryansh UI v9

## What changed

v9 changes the architecture from **precompute-all-scores then reveal them** to a **score stream consumer**.

The UI receives one result at a time:

```text
4-second window -> Yugal model -> score arrives -> graph updates -> next score
```

The graph remains empty until results arrive. It does not receive a complete score list at startup.

## Yugal integration contract

The preferred real-time interface is:

```python
def score_stream(wav_path):
    # model stays loaded
    # yields one score per 4-second window, in order
    yield score0
    yield score1
    yield score2
```

The UI consumes that generator continuously.

If Yugal instead exposes a function such as `score_window(window_16k_float32)`, wire it into the same streaming loop so one window produces one score.

Keep the older `score_file(wav_path) -> list[float]` interface for benchmark/offline evaluation if the team still needs it, but do not use it for the live UI path.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python test_suryansh.py
streamlit run app.py
```

## Important behavior

- 16 kHz mono normalization
- 4-second windows
- 0.5-second hop
- one score consumed at a time
- 5-window moving average
- 3-of-last-5 hysteresis
- 5-window warm-up
- raw dots + smoothed line
- no new graph points after score stream ends
