"""Pull a balanced slice of IndicSynth (Hugging Face) out to wav files.

IndicSynth is synthetic speech in 12 Indian languages, stored as parquet with the
audio bytes embedded in the rows. It is far too big to take whole, and taking it
whole would be a mistake anyway: one subset has 378k rows and another 34k, so an
even split matters more than volume.

WHY THIS DOES NOT USE datasets.load_dataset(streaming=True)
That path pulls an ENTIRE parquet row group before yielding the first row. With
audio inline a row group is hundreds of MB, so the script sits at
"<lang>: streaming ..." for many minutes with no output and looks hung. It also
routes decoding through torchcodec, which needs a matching FFmpeg build and fails
on Windows in ways that have nothing to do with our task.

Instead we download whole parquet shards (resumable, cached, with a progress bar)
and read the audio bytes straight out with pyarrow + soundfile. No torchcodec, no
streaming, no silent stalls.

    pip install -U huggingface_hub pyarrow soundfile
    python prep_indicsynth.py --list
    python prep_indicsynth.py --per-lang 1400 --out data/indicsynth
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = "vdivyasharma/IndicSynth"


def parquet_index():
    """-> {language: [parquet paths in the repo]}"""
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".parquet")]
    idx = defaultdict(list)
    for f in files:
        # paths look like "<Language>/train-00000-of-000NN.parquet" or
        # "data/<Language>/...". Take the first segment that is not "data".
        parts = [p for p in f.split("/")[:-1] if p.lower() != "data"]
        idx[parts[0] if parts else "unknown"].append(f)
    for k in idx:
        idx[k].sort()
    return dict(idx)


def audio_field(batch):
    """Find the struct column holding the encoded audio bytes."""
    for name in batch.schema.names:
        col = batch.column(name)
        try:
            fields = [f.name for f in col.type]
        except TypeError:
            continue
        if "bytes" in fields:
            return col.field("bytes")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show subsets, then exit")
    ap.add_argument("--langs", default="", help="comma-separated (default: all)")
    ap.add_argument("--per-lang", type=int, default=1400)
    ap.add_argument("--out", default="data/indicsynth")
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc}\n  pip install -U huggingface_hub pyarrow soundfile")

    try:
        idx = parquet_index()
    except Exception as exc:
        sys.exit(f"could not reach {REPO}: {exc}")

    if args.list:
        print(f"{len(idx)} subsets in {REPO}:\n")
        for k in sorted(idx):
            print(f"  {k:<14} {len(idx[k])} parquet shard(s)")
        return 0

    wanted = [w.strip() for w in args.langs.split(",") if w.strip()] or sorted(idx)
    missing = [w for w in wanted if w not in idx]
    if missing:
        print(f"  ! not in the repo, skipping: {', '.join(missing)}")
        wanted = [w for w in wanted if w in idx]
    if not wanted:
        sys.exit("no valid subsets -- run --list")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total, per_lang_counts = 0, {}

    for lang in wanted:
        print(f"\n{lang}: need {args.per_lang} clips")
        n = 0
        for shard in idx[lang]:
            if n >= args.per_lang:
                break
            print(f"  downloading {shard} ...")
            try:
                local = hf_hub_download(REPO, shard, repo_type="dataset")
            except Exception as exc:
                print(f"  ! shard failed ({exc}); trying the next one")
                continue

            try:
                pf = pq.ParquetFile(local)
                for batch in pf.iter_batches(batch_size=64):
                    if n >= args.per_lang:
                        break
                    col = audio_field(batch)
                    if col is None:
                        print(f"  ! no audio-bytes column in {shard}; columns: "
                              f"{batch.schema.names}")
                        break
                    for raw in col:
                        if n >= args.per_lang:
                            break
                        b = raw.as_py()
                        if not b:
                            continue
                        try:
                            w, sr = sf.read(io.BytesIO(b), dtype="float32",
                                            always_2d=False)
                        except Exception:
                            continue        # unreadable clip, skip quietly
                        if w.ndim > 1:
                            w = w.mean(axis=1)
                        if w.size < sr * 0.6:            # under ~0.6 s
                            continue
                        peak = float(np.max(np.abs(w))) or 1.0
                        w = (w / peak * 0.6).astype(np.float32)
                        sf.write(str(out / f"indicsynth_{lang}_{n:05d}.wav"), w, sr)
                        n += 1
                        if n % 200 == 0:
                            print(f"    {n}")
            except Exception as exc:
                print(f"  ! could not read {shard}: {exc}")

        print(f"  {lang}: {n} clips")
        per_lang_counts[lang] = n
        total += n

    print(f"\n{total} clips -> {out}")
    print("per language:")
    for k, v in sorted(per_lang_counts.items()):
        print(f"  {k:<14} {v}")

    print()
    print("NEXT -- extract, then stamp as SPOOF (label 1):")
    print(f"  python src/extract_embeddings.py --split train --audio-dir {out} \\")
    print(f"      --out outputs/embeddings_indicsynth --batch 8")
    print(f"  python stamp_labels.py --emb-dir outputs/embeddings_indicsynth/train --label 1")
    print(f"  python stamp_labels.py --emb-dir outputs/embeddings_indicsynth/train --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
