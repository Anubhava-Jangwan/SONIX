#!/usr/bin/env python3
"""
check_pipeline.py  --  SONIX / SIH26104

THE DECISIVE TEST: is the demo's scoring path consistent with the path that
produced our 1.49% EER?

Our benchmark number comes from  extract_embeddings.py -> eval.py.
The demo scores through a completely separate implementation, demo/score_file.py.
If those two compute embeddings differently, we would see exactly what we see:
a great benchmark number and everything scoring ~1.0 in the demo.

This takes clips with KNOWN labels straight from the ASVspoof protocol and runs
them through the DEMO path.

    cd demo
    python check_pipeline.py --n 12

HOW TO READ THE RESULT
  bonafide mean near 0 and spoof mean near 1
      -> the demo path is healthy. Real clips scoring 1.0 are then either
         genuinely out-of-domain, or actually AI audio.
  bonafide mean also near 1
      -> THE DEMO PATH IS BROKEN. The head is being fed embeddings that do not
         match what it was trained on. Fix this before touching anything else -
         no amount of augmentation or threshold tuning will help.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

import score_file as S

PROTOCOL_FILE = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
FLAC_DIR = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    ap.add_argument("--data-root", default="../data/asvspoof19_la")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=10, help="clips per class")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    root = Path(args.data_root)
    proto = root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILE[args.split]
    if not proto.exists():
        sys.exit(f"FATAL: protocol not found: {proto}\nPass --data-root")
    flac = root / FLAC_DIR[args.split] / "flac"

    bona, spoof = [], []
    for line in proto.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        (bona if p[4] == "bonafide" else spoof).append(p[1])

    random.seed(args.seed)
    bona = random.sample(bona, min(args.n, len(bona)))
    spoof = random.sample(spoof, min(args.n, len(spoof)))

    S.set_vad(enabled=False)            # raw model behaviour
    ckpt = S.resolve_ckpt(args.ckpt)
    print(f"model: {ckpt}")
    print(f"split: {args.split}   clips: {len(bona)} bonafide / {len(spoof)} spoof\n")

    out = {}
    for name, ids in (("bonafide", bona), ("spoof", spoof)):
        means = []
        print(f"--- {name} (expect {'LOW ~0' if name == 'bonafide' else 'HIGH ~1'}) ---")
        for i in ids:
            f = flac / f"{i}.flac"
            if not f.exists():
                print(f"  ! missing {f.name}")
                continue
            try:
                s = S.score_file(str(f), ckpt_path=ckpt)
            except Exception as exc:
                print(f"  ! {i}: {exc}")
                continue
            a = np.asarray(s, dtype=float)
            means.append(a.mean())
            print(f"  {i:<22} windows={len(s):<3} mean={a.mean():.4f} max={a.max():.4f}")
        out[name] = float(np.mean(means)) if means else float("nan")
        print()

    b, sp = out.get("bonafide", float("nan")), out.get("spoof", float("nan"))
    print("=" * 62)
    print(f"bonafide mean = {b:.4f}     spoof mean = {sp:.4f}")
    print("=" * 62)
    if not (b == b and sp == sp):
        print("VERDICT: could not score enough clips.")
    elif b < 0.30 and sp > 0.70:
        print("VERDICT: demo path looks HEALTHY - it separates known clips correctly.")
        print("         So a real clip scoring 1.0 is out-of-domain (or is AI audio).")
    elif b > 0.70:
        print("VERDICT: *** DEMO PATH IS BROKEN ***")
        print("         Known-bonafide training-domain clips score as FAKE through")
        print("         demo/score_file.py, even though eval.py gives 1.49% EER on")
        print("         this same data. The head is being fed embeddings that do not")
        print("         match training. FIX THIS FIRST - augmentation and threshold")
        print("         tuning cannot compensate for it.")
    else:
        print("VERDICT: separation is weak/unclear - send this output to the team.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
