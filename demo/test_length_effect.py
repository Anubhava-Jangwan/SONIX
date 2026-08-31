#!/usr/bin/env python3
"""
test_length_effect.py  --  SONIX / SIH26104

Is our real-world failure caused by WINDOW FULLNESS rather than by the voice?

Training audio (ASVspoof) is mostly 1-5 s, so most training windows are only
partly filled and the rest is zero-padded digital silence. Real recordings are
30-60 s, so every window is completely full. The feature extractor normalises
across the whole window, zeros included -- so padded and full windows land in
different regions of embedding space.

This scores ONE recording two ways:
  (a) full-length, normal windowing  -> completely FULL windows
  (b) short excerpts cut from the SAME audio -> ZERO-PADDED windows, like training

Same speaker, same mic, same room. The only thing that changes is window
fullness. So any difference in score is caused by that, not by the voice.

    cd demo
    python test_length_effect.py "C:\\path\\to\\my_voice.wav"

HOW TO READ IT
  full-length HIGH but short excerpts LOW
      -> window fullness / zero-padding is a real cause of our false alarms.
         Worth switching to repeat-padding (standard in RawNet2/AASIST recipes).
  both HIGH
      -> not a padding effect; it is ordinary domain shift (mic/room/channel),
         so RawBoost + recalibration is the right fix.
"""

import argparse
import sys

import numpy as np

import score_file as S

TARGET_SR = 16000
WIN = 64000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="a genuine (real human) recording, ideally 30s+")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--excerpt-secs", default="1,2,3",
                    help="excerpt lengths to test, comma separated")
    ap.add_argument("--n-excerpts", type=int, default=6,
                    help="excerpts sampled per length")
    ap.add_argument("--label", required=True, choices=["real", "fake"],
                    help="GROUND TRUTH for this file. Required: the verdict\n                         is meaningless without knowing what the clip IS.")
    args = ap.parse_args()

    S.set_vad(enabled=False)                 # raw behaviour, no gate
    ckpt = S.resolve_ckpt(args.ckpt)
    wav = S._load_audio(args.audio)
    dur = len(wav) / TARGET_SR
    print(f"model : {ckpt}")
    print(f"audio : {args.audio}")
    print(f"length: {dur:.1f} s\n")

    if dur < 8:
        print("! recording is short; 30 s or more makes this much clearer.\n")

    # (a) full-length, normal windowing -> FULL windows
    full = S.score_file(args.audio, ckpt_path=ckpt)
    a = np.asarray(full, dtype=float)
    n_full = sum(1 for _ in S._windows(wav))
    print("--- (a) FULL-LENGTH, normal windowing (windows are FULL) ---")
    print(f"  windows={n_full}  mean={a.mean():.4f}  median={np.median(a):.4f}  "
          f"max={a.max():.4f}  >=0.90: {(a >= 0.90).mean()*100:.0f}%\n")

    # (b)+(c) short excerpts, scored TWO ways from the SAME cut positions:
    #   (b) zero-padded by hand  = the OLD behaviour
    #   (c) through S._windows() = the PRODUCTION path (repeat-padded now)
    print("--- (b) OLD zero-padding   vs   (c) PRODUCTION path ---")
    rng = np.random.default_rng(1337)
    rows = []
    for secs in [float(s) for s in args.excerpt_secs.split(",")]:
        n = int(secs * TARGET_SR)
        if n >= len(wav):
            print(f"  {secs:.0f}s: recording too short to excerpt")
            continue
        old_means, new_means = [], []
        for _ in range(args.n_excerpts):
            st = int(rng.integers(0, len(wav) - n))
            chunk = wav[st:st + n]
            # (b) old: hand-built zero-padded window
            w = np.zeros(WIN, dtype=np.float32)
            w[:len(chunk)] = chunk
            old_means.append(S._score_windows([w], ckpt)[0])
            # (c) new: exactly what the demo does with this audio
            wins = list(S._windows(chunk))
            new_means.append(float(np.mean(S._score_windows(wins, ckpt))))
        om, nm = float(np.mean(old_means)), float(np.mean(new_means))
        rows.append((secs, om, nm))
        print(f"  {secs:.0f}s  zero-pad={om:.4f}   production={nm:.4f}   "
              f"{'IMPROVED' if nm < om - 0.02 else 'no change' if abs(nm-om) <= 0.02 else 'WORSE'}")

    print("\n" + "=" * 62)
    full_mean = float(a.mean())
    print(f"full-length mean = {full_mean:.4f}")
    for secs, om, nm in rows:
        print(f"{secs:.0f}s  old zero-pad = {om:.4f}   production = {nm:.4f}")
    print("=" * 62)

    print(f"ground truth: {args.label.upper()}")
    print("=" * 62)

    if not rows:
        return 0
    best = min(nm for _, _, nm in rows)
    old_best = min(om for _, om, _ in rows)
    gain = old_best - best

    if gain > 0.02:
        print(f"PADDING FIX IS WORKING: short clips improved by {gain:.3f} "
              f"({old_best:.3f} -> {best:.3f}).")
    elif gain < -0.02:
        print(f"WARNING: the padding change made short clips WORSE "
              f"({old_best:.3f} -> {best:.3f}). Tell me.")
    else:
        print("Padding change made little difference on this clip.")
    print()

    if args.label == "fake":
        if full_mean > 0.70 and best > 0.70:
            print("VERDICT: CORRECT. Detected as fake at every length, padded or")
            print("  full. Nothing to fix here - and this is a good demo clip.")
        elif full_mean > 0.70:
            print("VERDICT: caught at full length but MISSED in short excerpts.")
            print("  Short calls would evade us - worth reporting to the team.")
        else:
            print("VERDICT: MISSED. A known-fake clip is scoring real. Investigate.")
        print("\n  NOTE: this file is fake, so it says NOTHING about our")
        print("  false-alarm problem. For that, run a GENUINE recording with")
        print("  --label real.")
        return 0

    if full_mean > 0.60 and best < 0.30:
        print("VERDICT: WINDOW FULLNESS IS IMPLICATED.")
        print("  The same voice scores REAL in short padded excerpts but FAKE")
        print("  at full length. That is a padding/length artefact, not the")
        print("  voice. Fix: repeat-padding instead of zero-padding, applied")
        print("  consistently in extraction AND scoring.")
    elif full_mean > 0.60 and best > 0.60:
        print("VERDICT: NOT a padding effect - it scores fake either way.")
        print("  This is ordinary domain shift (mic / room / channel).")
        print("  RawBoost + threshold recalibration is the right fix.")
    elif full_mean < 0.30:
        print("VERDICT: this recording scores REAL at full length - good.")
        print("  Our false alarms are not universal; collect more clips to")
        print("  find which ones actually fail.")
    else:
        print("VERDICT: mixed / borderline - send this output to the team.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
