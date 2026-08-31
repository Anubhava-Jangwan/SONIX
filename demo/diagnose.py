#!/usr/bin/env python3
"""
diagnose.py -- per-window energy + spoof score for one clip (both models).

    cd demo
    python diagnose.py "path\\to\\real_clip.wav"

Tells us WHY a real clip gets flagged: if the high scores line up with
low-energy (silent) windows, a VAD/energy gate fixes it. If speech windows
themselves score high, it's pure domain shift (needs recalibration / more real
bonafide data). Uses the demo's own score_file pipeline, so numbers match the UI.
"""
import sys
import numpy as np

import score_file as S


def rms_dbfs(w):
    r = float(np.sqrt(np.mean(np.square(w)) + 1e-12))
    return 20.0 * np.log10(r + 1e-12)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python diagnose.py <clip.wav> [--vad-db -45]")
    path = sys.argv[1]
    vad_db = -45.0
    if "--vad-db" in sys.argv:
        vad_db = float(sys.argv[sys.argv.index("--vad-db") + 1])

    ckpts = ["outputs/models/head.pt"]
    if S.checkpoint_available("outputs/models/head_aug.pt"):
        ckpts.append("outputs/models/head_aug.pt")

    wav = S._load_audio(path)
    wins = list(S._windows(wav))
    dbs = np.array([rms_dbfs(w) for w in wins])

    scores = {c: [] for c in ckpts}
    for c in ckpts:
        for i in range(0, len(wins), 8):
            scores[c].extend(S._score_windows(wins[i:i + 8], c))
    for c in ckpts:
        scores[c] = np.array(scores[c])

    names = [c.split("/")[-1] for c in ckpts]
    print(f"\n{path}")
    print(f"{len(wins)} windows · VAD line at {vad_db:.0f} dBFS\n")
    print("win  start   dBFS  speech  " + "  ".join(f"{n:>10}" for n in names))
    for i, w in enumerate(wins):
        sp = "yes" if dbs[i] > vad_db else " . "
        row = f"{i:3d}  {i*0.5:5.1f}  {dbs[i]:6.1f}   {sp}   " + \
              "  ".join(f"{scores[c][i]:10.3f}" for c in ckpts)
        print(row)

    speech = dbs > vad_db
    print("\n--- summary (label: real clip should score LOW / near 0) ---")
    for c, n in zip(ckpts, names):
        a = scores[c]
        sm = a[speech].mean() if speech.any() else float("nan")
        qm = a[~speech].mean() if (~speech).any() else float("nan")
        print(f"[{n}] all: mean={a.mean():.3f} max={a.max():.3f} | "
              f"speech-win mean={sm:.3f} | silence-win mean={qm:.3f} | "
              f">=0.10: {(a >= 0.10).mean()*100:.0f}%  >=0.90: {(a >= 0.90).mean()*100:.0f}%")
    print(f"\n{speech.sum()}/{len(wins)} windows are speech (> {vad_db:.0f} dBFS); "
          f"{(~speech).sum()} are silence/low.\n")


if __name__ == "__main__":
    main()
