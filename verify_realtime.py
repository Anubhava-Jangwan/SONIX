#!/usr/bin/env python3
"""
verify_realtime.py -- does the LIVE path agree with the DEMO path?

realtime/ and demo/ are two separate implementations of the same pipeline. If
they disagree, the live demo shows different numbers from every result we have
measured and reported. This scores the same audio through both and compares.

    python verify_realtime.py "data/test_clips/suryansh_voice.wav"
"""
import sys, numpy as np, torch

WIN, HOP, SR = 64000, 8000, 16000

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/test_clips/suryansh_voice.wav"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "outputs/models/head.pt"

    sys.path.insert(0, "demo")
    import score_file as S
    from realtime import frontend
    from realtime.checkpoint import load_checkpoint

    wav = S._load_audio(path)
    wins = [w for w in S._windows(wav)][:8]      # first 8 windows is plenty
    print(f"clip: {path}\nwindows compared: {len(wins)}\n")

    # --- demo path (the one all our reported numbers came from) ---
    S.set_vad(enabled=False)
    demo_scores = S._score_windows(wins, ckpt)

    # --- live path ---
    model, cfg = load_checkpoint(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    emb = frontend.embed(wins)
    dev = next(model.parameters()).device
    with torch.no_grad():
        live_scores = torch.sigmoid(
            model(torch.from_numpy(emb).float().to(dev)).squeeze(-1)).cpu().numpy()

    print(f"{'win':>4} {'demo':>9} {'live':>9} {'diff':>9}")
    diffs = []
    for i, (a, b) in enumerate(zip(demo_scores, live_scores)):
        diffs.append(abs(a - b))
        print(f"{i:>4} {a:9.4f} {float(b):9.4f} {abs(a-float(b)):9.4f}")

    m = max(diffs)
    print(f"\nmax difference: {m:.4f}")
    if m < 0.02:
        print("PASS - the live path matches the demo path. Same model, same numbers.")
    elif m < 0.10:
        print("CLOSE - small gap, most likely fp16 vs fp32. Acceptable, but note it.")
    else:
        print("*** FAIL - the two paths disagree. Do NOT demo live until this is fixed.")
        print("    Most likely cause: mu/sd standardisation missing on one side.")

if __name__ == "__main__":
    main()
