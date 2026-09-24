"""Score every registered head over a folder of labelled clips and report, per head
and per group, whether the model can RANK real vs fake at all.

The question this answers is not "does it get the verdict right". It is:

    Is there signal that the threshold is hiding, or is there no signal?

Those look identical on a green/red badge and completely different in the numbers.
AUC ~= 0.5 means the model cannot separate the two classes no matter where you put
the threshold. AUC >> 0.5 with bad verdicts means the model CAN separate them and
the operating point is simply in the wrong place for this domain -- which is a
much cheaper problem.

The embedding pass is the expensive part and it does not depend on the head, so we
embed each clip ONCE and score all heads on the cached vectors. That is the whole
point of the frozen-front-end design, and it makes this run in minutes.

    python diagnose_pairs.py --dir sonix_real/sonix_real
    python diagnose_pairs.py --dir sonix_real/sonix_real --extra-real data/heldout/real --extra-fake data/heldout/mms_tts

Layout expected under --dir:  real/*.wav  (label 0)   fake/*.wav  (label 1)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

WIN = 64000          # 4.0 s @ 16 kHz
HOP = 8000           # 0.5 s
SR = 16000

# ───────────────────────────────────────────────────────────── audio


def load_mono16k(path):
    """Decode to float32 mono @ 16 kHz. Repeat-pad anything shorter than a window.

    Repeat-pad, never zero-pad: padding with digital silence is the bug that made
    short genuine clips score 0.88 instead of 0.03.
    """
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if sr != SR:
        from realtime.resample import resample
        mono = resample(mono, sr, SR)
    if mono.size < WIN:
        reps = int(np.ceil(WIN / max(1, mono.size)))
        mono = np.tile(mono, reps)[:WIN].astype(np.float32)
    return mono


def windows_of(sig):
    return [sig[s:s + WIN] for s in range(0, sig.size - WIN + 1, HOP)]


def speech_mask(wins, frame=400):
    """The live path's adaptive silence gate, so these numbers match the dashboard.

    Floor scales to the clip: 12% of its 90th-percentile frame RMS, clipped into a
    sane band. A fixed floor rejects a quiet recording wholesale.
    """
    keep = []
    for w in wins:
        n = w.size // frame
        if n == 0:
            keep.append(True)
            continue
        rms = np.sqrt(np.mean(np.square(w[:n * frame].reshape(n, frame).astype(np.float64)), axis=1))
        floor = float(np.clip(np.percentile(rms, 90) * 0.12, 0.0006, 0.01))
        keep.append(bool(np.mean(rms > floor) >= 0.20))
    return np.array(keep, dtype=bool)


# ───────────────────────────────────────────────────────────── metrics


def auc(pos, neg):
    """P(a random fake scores above a random real). Rank-based, ties at 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rpos = ranks[:pos.size].sum()
    return float((rpos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def eer(pos, neg):
    """Equal error rate over the pooled score axis."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ths = np.unique(np.concatenate([pos, neg]))
    best, val = 1.0, 1.0
    for t in ths:
        far = float(np.mean(neg >= t))      # real called fake
        frr = float(np.mean(pos < t))       # fake called real
        if abs(far - frr) < best:
            best, val = abs(far - frr), (far + frr) / 2
    return val


# ───────────────────────────────────────────────────────────── main


def group_of(name):
    """Named celebrity clips and the numbered set are different provenance.

    The numbered pairs are team phone recordings vs free-tool clones; the named
    ones are broadcast audio vs free-tool clones of public figures. Mixing them
    hides which one the model is failing on.
    """
    stem = Path(name).stem.lower()
    if stem.startswith("real_") or stem.startswith("fake_"):
        return "numbered"
    return "named"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder containing real/ and fake/")
    ap.add_argument("--extra-real", action="append", default=[],
                    help="extra bonafide folder (repeatable), grouped by folder name")
    ap.add_argument("--extra-fake", action="append", default=[],
                    help="extra spoof folder (repeatable) -- point this at held-out "
                         "MMS-TTS / IndicSynth to test a SEEN generator family")
    ap.add_argument("--models", default="", help="comma-separated keys; default = all present")
    ap.add_argument("--agg", default="mean", choices=["mean", "median", "max", "p90"],
                    help="how to reduce per-window scores to one clip score")
    ap.add_argument("--no-vad", action="store_true", help="score every window, gate off")
    ap.add_argument("--batch", type=int, default=8,
                    help="windows per front-end forward pass. A 116-window clip "
                         "sent in one go needs ~4 GB and OOMs a 6 GB card; drop "
                         "to 4 or 2 if it still fails.")
    ap.add_argument("--device", default=None,
                    help="force 'cuda' or 'cpu' (default: auto-detect)")
    ap.add_argument("--out", default="outputs/diagnose_pairs.csv")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from realtime import models as registry
    from realtime.checkpoint import load_checkpoint
    from realtime import frontend
    import torch

    # ---- which heads
    cat = [m for m in registry.catalogue() if m["exists"]]
    if args.models:
        want = {k.strip() for k in args.models.split(",")}
        cat = [m for m in cat if m["key"] in want]
    if not cat:
        sys.exit("no checkpoints found under outputs/models/")
    print(f"heads: {', '.join(m['key'] for m in cat)}\n")

    # ---- which clips
    root = Path(args.dir)
    clips = []   # (path, label, group)
    for sub, lab in (("real", 0), ("fake", 1)):
        d = root / sub
        if not d.is_dir():
            sys.exit(f"missing {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
                clips.append((p, lab, group_of(p.name)))
    for d in args.extra_real:
        for p in sorted(Path(d).glob("*.wav")):
            clips.append((p, 0, Path(d).name))
    for d in args.extra_fake:
        for p in sorted(Path(d).glob("*.wav")):
            clips.append((p, 1, Path(d).name))
    if not clips:
        sys.exit("no audio found")
    print(f"{len(clips)} clips\n")

    # ---- embed once, reuse for every head
    print("loading front-end ...")
    frontend.load()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    embs, meta = [], []
    for i, (p, lab, grp) in enumerate(clips, 1):
        try:
            sig = load_mono16k(p)
        except Exception as exc:
            print(f"  !! {p.name}: {exc}")
            continue
        wins = windows_of(sig)
        if not wins:
            print(f"  !! {p.name}: no full window")
            continue
        kept = np.ones(len(wins), bool) if args.no_vad else speech_mask(wins)
        if not kept.any():
            print(f"  !! {p.name}: every window failed the gate")
            continue
        use = [w for w, k in zip(wins, kept) if k]
        # Chunked: frontend.embed(use) on a whole clip allocates
        # len(use) x 64000 samples through the conv stack at once, which is
        # ~3.9 GB for a 116-window clip and OOMs a 6 GB card partway through
        # the set. Same result, bounded peak memory.
        e = np.concatenate(
            [frontend.embed(use[j:j + args.batch])
             for j in range(0, len(use), args.batch)], axis=0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        embs.append(e)
        meta.append({"file": p.name, "label": lab, "group": grp,
                     "windows": len(wins), "scored": int(kept.sum())})
        print(f"  [{i}/{len(clips)}] {p.name:44s} {kept.sum():4d}/{len(wins):4d} windows")

    # ---- score every head on the cached embeddings
    rows = []
    for m in cat:
        head, _cfg = load_checkpoint(m["resolved_path"], device)
        head.eval()
        for e, info in zip(embs, meta):
            with torch.no_grad():
                xb = torch.from_numpy(np.ascontiguousarray(e)).float().to(device)
                logit = head(xb)
                s = (torch.softmax(logit, -1)[:, 1] if logit.shape[-1] == 2
                     else torch.sigmoid(logit).squeeze(-1))
                s = s.cpu().numpy().reshape(-1)
            agg = {"mean": np.mean, "median": np.median, "max": np.max,
                   "p90": lambda v: np.percentile(v, 90)}[args.agg](s)
            rows.append({"model": m["key"], **info, "score": float(agg)})
        print(f"scored: {m['key']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- the table that decides everything
    groups = sorted({r["group"] for r in rows})
    print("\n" + "=" * 96)
    print(f"{'model':<18}{'group':<14}{'n_real':>7}{'n_fake':>7}{'AUC':>8}{'EER':>8}"
          f"{'med real':>10}{'med fake':>10}{'verdict':>12}")
    print("-" * 96)
    for m in cat:
        for g in groups:
            sel = [r for r in rows if r["model"] == m["key"] and r["group"] == g]
            pos = [r["score"] for r in sel if r["label"] == 1]
            neg = [r["score"] for r in sel if r["label"] == 0]
            if not pos or not neg:
                continue
            a, e_ = auc(pos, neg), eer(pos, neg)
            if a != a:
                verdict = "-"
            elif a >= 0.85:
                verdict = "STRONG"
            elif a >= 0.65:
                verdict = "signal"
            elif a >= 0.55:
                verdict = "weak"
            else:
                verdict = "NO SIGNAL"
            print(f"{m['key']:<18}{g:<14}{len(neg):>7}{len(pos):>7}{a:>8.3f}{e_:>8.3f}"
                  f"{np.median(neg):>10.4f}{np.median(pos):>10.4f}{verdict:>12}")
    print("=" * 96)
    print("""
HOW TO READ THIS

  AUC ~ 0.50   The head cannot rank real above fake in this group. No threshold,
               no ensemble and no router can recover it -- there is nothing to
               recover. The fix has to be in the training data.

  AUC > 0.65   The head CAN rank them. If the verdicts are still wrong, the
               operating point is wrong for this domain, not the model. Compare
               'med real' and 'med fake' against your amber (0.10) and red (0.90)
               thresholds -- if both medians sit on the same side of 0.10 but are
               well separated from each other, you have been thresholding away a
               working detector.

  Compare groups. High AUC on a SEEN generator family (extra-fake pointed at
  held-out MMS-TTS / IndicSynth) plus ~0.50 on the free-tool clones means a
  generator-generalisation gap, not a language shortcut -- a different problem
  with a different fix.
""")
    print(f"per-clip scores: {args.out}")


if __name__ == "__main__":
    main()
