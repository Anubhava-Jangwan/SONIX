#!/usr/bin/env python3
"""
eval.py  --  SONIX / SIH26104

Score a cached split with the trained head, save the two arrays Yukti needs, and
compute EER as a cross-check. This produces THE number of the whole project.

    T8  in-domain:   python src/eval.py --split eval --model-ckpt outputs/models/head.pt
    T9  cross-data:  python src/eval.py --split itw  --model-ckpt outputs/models/head.pt

For Yukti it writes  scores/<split>_labels.npy  and  scores/<split>_scores.npy
    labels: 1 = spoof
    scores: higher = more likely spoof (calibrated probability via sigmoid)
She loads those into metrics.py without ever touching this file. Our EER here and
hers should agree; if they don't, one of us has a bug -- find it before the slide.

SANITY (printed automatically):
    low single digits  -> working
    ~40%               -> a bug, almost always the protocol parse or a label flip
    exactly 0%         -> eval leaked into training
In-the-Wild WILL be dramatically worse than eval. That gap is the headline, not a
bug -- record both numbers side by side.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def eer(labels, scores):
    """Returns (eer, threshold). labels: 1 = spoof. Local cross-check of Yukti's."""
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])


def load_split(emb_root, split):
    d = Path(emb_root) / split
    shards = sorted(s for s in d.glob("shard_*.npy") if ".labels" not in s.name)
    if not shards:
        sys.exit(f"FATAL: no shards in {d}. Run extract_embeddings.py --split "
                 f"{split} first.")
    Xs, ys = [], []
    for s in shards:
        lab = s.with_suffix(".labels.npy")
        if not lab.exists():
            sys.exit(f"FATAL: {s} has no matching {lab.name}. Re-extract this "
                     f"shard -- embeddings and labels must stay paired.")
        emb, y = np.load(s), np.load(lab)
        if len(emb) != len(y):
            sys.exit(f"FATAL: {s.name} rows {len(emb)} != labels {len(y)}.")
        Xs.append(emb.astype(np.float32))
        ys.append(y.astype(np.int64))
    X, y = np.concatenate(Xs, 0), np.concatenate(ys, 0)
    print(f"[{split}] {len(X)} vectors  (spoof={int((y==1).sum())}, "
          f"bonafide={int((y==0).sum())})")
    return X, y


def build_head(cfg):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(cfg["in_dim"], cfg["hidden"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden"], 1),
    )


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description="Score a split and compute EER.")
    ap.add_argument("--split", required=True,
                    help="eval, dev, train, or itw (any extracted split)")
    ap.add_argument("--emb-root", default="outputs/embeddings")
    ap.add_argument("--model-ckpt", default="outputs/models/head.pt")
    ap.add_argument("--out-scores", default="outputs/scores")
    ap.add_argument("--batch", type=int, default=4096,
                    help="scoring mini-batch (lower it if the GPU runs out)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    ckpt_p = Path(args.model_ckpt)
    if not ckpt_p.exists():
        sys.exit(f"FATAL: checkpoint {ckpt_p} not found. Run train.py first.")
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_head(ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    mu = np.array(ckpt["mu"], np.float32)
    sd = np.array(ckpt["sd"], np.float32)
    sd[sd == 0] = 1.0            # a constant feature would otherwise give inf/nan

    X, y = load_split(args.emb_root, args.split)
    in_dim = int(ckpt["config"]["in_dim"])
    if X.shape[1] != in_dim:
        sys.exit(f"FATAL: shards are {X.shape[1]}-dim but the checkpoint expects "
                 f"{in_dim}. These embeddings came from a different front-end.")
    X = (X - mu) / sd

    # score in chunks -- eval is ~71k x 1024, too big to push onto a 4-6 GB card
    # (or a laptop's RAM) as one tensor
    scores = np.empty(len(X), np.float32)
    logits_all = np.empty(len(X), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            xb = torch.from_numpy(X[i:i + args.batch]).float().to(device)
            lg = model(xb).squeeze(1)
            # keep every chunk's logits -- `lg` alone would only be the LAST
            # chunk, which would silently save a truncated logits array
            logits_all[i:i + args.batch] = lg.cpu().numpy()
            scores[i:i + args.batch] = torch.sigmoid(lg).cpu().numpy()
    # scores: higher = more spoof

    out = Path(args.out_scores)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{args.split}_labels.npy", y.astype(np.int64))
    np.save(out / f"{args.split}_scores.npy", scores.astype(np.float32))
    # Raw logits too: sigmoid() saturates to exactly 1.0 in float32, so the
    # logits CANNOT be recovered from the scores afterwards. Calibration
    # (temperature scaling) needs them, so they are saved at source.
    np.save(out / f"{args.split}_logits.npy", logits_all.astype(np.float32))
    print(f"saved -> {out / (args.split + '_labels.npy')}  ,  "
          f"{args.split}_scores.npy  and  {args.split}_logits.npy"
          f"  (hand these to Yukti)")

    e, thr = eer(y, scores)
    print("=" * 56)
    print(f"  {args.split.upper()} EER = {e * 100:.2f}%   (threshold {thr:.4f})")
    print("=" * 56)

    # sanity guidance
    pct = e * 100
    if np.isnan(pct):
        print("  ! only one class present -- check the labels.")
    elif args.split == "itw":
        print("  In-the-Wild is expected to be far higher than eval. This is the "
              "cross-dataset generalisation gap -- your headline finding.")
    elif pct < 0.01:
        print("  ! EER ~ 0%. Suspicious: eval may have leaked into training. "
              "Confirm train/dev/eval shards came from different splits.")
    elif pct < 8:
        print("  low single digits -- in-domain detector is working as expected.")
    elif 30 < pct < 50 and args.split in ("eval", "dev", "train"):
        print("  ! ~40% on an in-domain split means a bug -- almost always the "
              "protocol parse or a flipped label. Re-run verify_protocol.py.")
    else:
        print("  Note this number and compare with Yukti's metrics.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
