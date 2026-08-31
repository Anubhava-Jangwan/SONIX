#!/usr/bin/env python3
"""
make_demo_scores.py  --  SONIX / SIH26104

Precompute the score stream for every demo clip, so the live demo REPLAYS
validated numbers instead of running inference on stage.

WHY THIS EXISTS
The model currently misfires on out-of-domain real audio. Running it live, on
unpredictable walk-up audio, in front of a jury, is an avoidable risk. We score
our curated clips here, once, check they behave, and the UI just replays them.

    cd demo
    python make_demo_scores.py --clips-dir "..\\data\\demo_clips" --ckpt outputs/models/head.pt

Writes one JSON per clip into --out-dir (default: demo/scores):

    { "clip": "real_01.wav", "model": "...head.pt", "win_s": 4.0, "hop_s": 0.5,
      "vad": {...}, "scores": [0.02, 0.03, ...] }

plus an index.json listing every clip with its final risk band, so Suryansh's
clip-library dropdown can read one file.

It also prints a per-clip verdict using the SAME smoothing + hysteresis as the
UI, so you can see immediately which clips are safe to demo. Resumable: existing
JSONs are skipped unless --force.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import score_file as S
from risk import hysteresis_bands, moving_average


def final_band(scores, amber, red):
    """Same smoothing + hysteresis the UI applies, so the verdict matches."""
    if not scores:
        return "GREEN"
    sm = moving_average(scores, window=5)
    bands = hysteresis_bands(sm, amber_threshold=amber, red_threshold=red,
                             agree_count=3, history_size=5,
                             initial_band="GREEN", warmup_windows=5)
    return bands[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute demo score streams.")
    ap.add_argument("--clips-dir", required=True,
                    help="folder of .wav/.flac demo clips")
    ap.add_argument("--out-dir", default="scores")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (default: baseline outputs/models/head.pt)")
    ap.add_argument("--amber", type=float, default=0.10)
    ap.add_argument("--red", type=float, default=0.90)
    ap.add_argument("--no-vad", action="store_true",
                    help="disable the silence gate (it is ON by default)")
    ap.add_argument("--vad-db", type=float, default=-45.0)
    ap.add_argument("--force", action="store_true", help="rescore existing JSONs")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.is_dir():
        sys.exit(f"FATAL: --clips-dir not found: {clips_dir.resolve()}")
    clips = sorted(list(clips_dir.glob("*.wav")) + list(clips_dir.glob("*.flac")))
    if not clips:
        sys.exit(f"FATAL: no .wav/.flac in {clips_dir.resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    S.set_vad(enabled=not args.no_vad, dbfs=args.vad_db)
    try:
        ckpt = S.resolve_ckpt(args.ckpt)
    except FileNotFoundError as exc:
        sys.exit(f"FATAL: {exc}")

    print(f"model : {ckpt}")
    print(f"vad   : {S.get_vad()}")
    print(f"bands : amber {args.amber}  red {args.red}")
    print(f"clips : {len(clips)} in {clips_dir}\n")

    index = []
    for c in clips:
        dst = out_dir / f"{c.stem}.json"
        if dst.exists() and not args.force:
            try:
                scores = json.loads(dst.read_text())["scores"]
                band = final_band(scores, args.amber, args.red)
                index.append({"clip": c.name, "scores_file": dst.name,
                              "windows": len(scores), "final_band": band})
                print(f"  = {c.name:<34} (already scored)  {band}")
                continue
            except Exception:
                pass  # unreadable -> rescore

        try:
            scores = S.score_file(str(c), ckpt_path=ckpt)
        except Exception as exc:
            print(f"  ! {c.name:<34} FAILED: {exc}")
            continue

        band = final_band(scores, args.amber, args.red)
        dst.write_text(json.dumps({
            "clip": c.name,
            "model": str(ckpt),
            "win_s": 4.0,
            "hop_s": 0.5,
            "vad": S.get_vad(),
            "generated": datetime.now().isoformat(timespec="seconds"),
            "scores": [round(float(s), 6) for s in scores],
        }, indent=1))

        a = np.asarray(scores, dtype=float)
        index.append({"clip": c.name, "scores_file": dst.name,
                      "windows": len(scores), "final_band": band})
        print(f"  + {c.name:<34} n={len(scores):<4} "
              f"mean={a.mean():.3f} max={a.max():.3f}  -> {band}")

    (out_dir / "index.json").write_text(json.dumps({
        "model": str(ckpt),
        "amber": args.amber, "red": args.red,
        "vad": S.get_vad(),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "clips": index,
    }, indent=1))

    print(f"\nwrote {len(index)} score files + index.json -> {out_dir.resolve()}")
    print("Hand this folder to Suryansh for the demo's Replay mode.")
    print("\nSanity check before you trust a clip on stage:")
    print("  real clips should end GREEN; cloned clips should end RED.")
    print("  Anything that disagrees is NOT demo-safe - investigate, do not ship it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
