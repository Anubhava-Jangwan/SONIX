"""Pull a balanced slice of IndicSynth (Hugging Face) out to wav files.

IndicSynth is synthetic speech in 12 Indian languages, stored as parquet with the
audio inline. It is far too big to take whole, and taking it whole would be a
mistake anyway: one language with 378k rows would drown the rest and the head
would learn that language instead of learning synthesis.

This takes an even slice per language and writes plain wavs that
extract_embeddings.py --audio-dir can read.

    pip install -U datasets soundfile
    python prep_indicsynth.py --list
    python prep_indicsynth.py --per-lang 1400 --out data/indicsynth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = "vdivyasharma/IndicSynth"
TARGET_SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show language subsets, then exit")
    ap.add_argument("--langs", default="", help="comma-separated subsets (default: all)")
    ap.add_argument("--per-lang", type=int, default=1400, help="clips per language")
    ap.add_argument("--out", default="data/indicsynth")
    args = ap.parse_args()

    try:
        import soundfile as sf
        from datasets import load_dataset, get_dataset_config_names
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc}\n  pip install -U datasets soundfile")

    try:
        configs = get_dataset_config_names(REPO)
    except Exception as exc:
        sys.exit(f"could not reach {REPO}: {exc}")

    if args.list:
        print(f"{len(configs)} subsets in {REPO}:\n")
        for c in configs:
            print(f"  {c}")
        return 0

    wanted = [w.strip() for w in args.langs.split(",") if w.strip()] or configs
    missing = [w for w in wanted if w not in configs]
    if missing:
        print(f"  ! not in the repo, skipping: {', '.join(missing)}")
        wanted = [w for w in wanted if w in configs]
    if not wanted:
        sys.exit("no valid subsets requested -- run --list")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0

    for lang in wanted:
        print(f"\n{lang}: streaming ...")
        try:
            # streaming avoids pulling a whole language's parquet to disk when we
            # only want a slice of it
            ds = load_dataset(REPO, lang, split="train", streaming=True)
        except Exception as exc:
            print(f"  ! skipped ({exc})")
            continue

        n = 0
        for row in ds:
            if n >= args.per_lang:
                break
            aud = row.get("audio")
            if not isinstance(aud, dict) or "array" not in aud:
                continue
            w = np.asarray(aud["array"], dtype=np.float32).reshape(-1)
            sr = int(aud.get("sampling_rate") or TARGET_SR)
            if w.size < sr * 0.6:                    # drop anything under ~0.6s
                continue
            peak = float(np.max(np.abs(w))) or 1.0
            w = (w / peak * 0.6).astype(np.float32)
            sf.write(str(out / f"indicsynth_{lang}_{n:05d}.wav"), w, sr)
            n += 1
            if n % 200 == 0:
                print(f"  {n}")
        print(f"  {lang}: {n} clips")
        total += n

    print(f"\n{total} clips -> {out}")
    print()
    print("NEXT -- extract, then stamp as SPOOF (label 1):")
    print(f"  python src/extract_embeddings.py --split train --audio-dir {out} \\")
    print(f"      --out outputs/embeddings_indicsynth --batch 8")
    print(f"  python stamp_labels.py --emb-dir outputs/embeddings_indicsynth/train --label 1")
    print(f"  python stamp_labels.py --emb-dir outputs/embeddings_indicsynth/train --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
