#!/usr/bin/env python3
"""
make_codec.py  --  SONIX / SIH26104

Make a codec-degraded copy of an ASVspoof split, so we can measure how the
detector holds up under real phone-call compression. This is EVALUATION support,
not retraining: you run the SAME trained model on this degraded copy and compare
the EER to the clean baseline.

What it does: passes each clip through the G.711 codec (mu-law at 8 kHz -- the
codec the landline telephone network actually uses), then back to 16 kHz FLAC, so
the compression artifacts are baked in. It mirrors the ASVspoof folder layout and
writes a matching protocol, so your existing extract_embeddings.py runs on it
unchanged.

USAGE
    # 1) make a codec-degraded copy of (a subset of) the eval split
    python make_codec.py --split eval --data-root data/asvspoof19_la \
                         --out data/asvspoof19_la_g711 --limit 5000

    # 2) extract embeddings from the degraded copy (your normal script)
    python src/extract_embeddings.py --split eval --batch 4 \
                         --data-root data/asvspoof19_la_g711 \
                         --out outputs/embeddings_g711

    # 3) Yugal scores it against the SAME head.pt and compares to the clean EER
    python src/eval.py --split eval --emb-root outputs/embeddings_g711 \
                         --out-scores outputs/scores_g711

--limit 0 processes the whole split. Resumable: already-converted files are
skipped, so re-run if it stops. One bad file is logged and skipped, never fatal.

Needs ffmpeg on PATH (https://ffmpeg.org) and soundfile.
"""

import argparse
import subprocess
import sys
import tempfile
import os
from pathlib import Path

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


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def transcode_g711(src: str, dst: str) -> None:
    """FLAC -> G.711 mu-law @ 8 kHz -> FLAC @ 16 kHz (artifacts baked in)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = tf.name
    try:
        # stage 1: downsample to 8 kHz mono and encode as mu-law (the codec step)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", tmp],
            check=True, capture_output=True)
        # stage 2: decode + resample back to 16 kHz, store as FLAC
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
             "-ar", "16000", "-ac", "1", "-c:a", "flac", dst],
            check=True, capture_output=True)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Codec-degrade an ASVspoof split.")
    ap.add_argument("--split", required=True, choices=["train", "dev", "eval"])
    ap.add_argument("--data-root", default="data/asvspoof19_la")
    ap.add_argument("--out", required=True,
                    help="output data-root for the codec'd copy")
    ap.add_argument("--codec", default="g711", choices=["g711"],
                    help="phone codec to simulate (g711 = mu-law 8kHz)")
    ap.add_argument("--limit", type=int, default=5000,
                    help="only convert the first N clips (0 = whole split)")
    args = ap.parse_args()

    if not have_ffmpeg():
        sys.exit("FATAL: ffmpeg not found on PATH. Install it from ffmpeg.org "
                 "(Windows: winget install Gyan.FFmpeg) and reopen the terminal.")

    src_root = Path(args.data_root)
    proto = src_root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILE[args.split]
    if not proto.exists():
        sys.exit(f"FATAL: protocol not found: {proto}")
    src_flac = src_root / FLAC_DIR[args.split] / "flac"

    out_root = Path(args.out)
    out_flac = out_root / FLAC_DIR[args.split] / "flac"
    out_proto_dir = out_root / "ASVspoof2019_LA_cm_protocols"
    out_flac.mkdir(parents=True, exist_ok=True)
    out_proto_dir.mkdir(parents=True, exist_ok=True)

    # read protocol lines, keep the subset, sort for reproducibility
    lines = [ln for ln in proto.read_text().splitlines() if ln.strip()]
    lines.sort(key=lambda ln: ln.split()[1] if len(ln.split()) >= 2 else ln)
    if args.limit:
        lines = lines[: args.limit]

    # write the matching (subset) protocol so extract_embeddings only sees these
    (out_proto_dir / PROTOCOL_FILE[args.split]).write_text("\n".join(lines) + "\n")

    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **k):
            return x

    done = skipped = failed = 0
    for ln in tqdm(lines, desc=f"{args.codec}:{args.split}", unit="file"):
        parts = ln.split()
        if len(parts) < 5:
            continue
        name = parts[1]
        src = src_flac / f"{name}.flac"
        dst = out_flac / f"{name}.flac"
        if dst.exists():
            skipped += 1
            continue
        if not src.exists():
            print(f"\n  ! missing source {src}")
            failed += 1
            continue
        try:
            transcode_g711(str(src), str(dst))
            done += 1
        except subprocess.CalledProcessError as e:
            print(f"\n  ! ffmpeg failed on {name}: {e.stderr.decode()[:200]}")
            failed += 1

    print(f"\n[{args.codec}/{args.split}] converted={done}  already={skipped}  "
          f"failed={failed}  total={len(lines)}")
    print(f"codec'd data-root: {out_root.resolve()}")
    print(f"next: python src/extract_embeddings.py --split {args.split} "
          f"--data-root {args.out} --out outputs/embeddings_{args.codec} --batch 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
