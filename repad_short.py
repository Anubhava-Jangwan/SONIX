#!/usr/bin/env python3
"""
repad_short.py  --  SONIX / SIH26104

Convert an existing embedding root from zero-padding to repeat-padding IN PLACE,
re-embedding only the clips shorter than 4.0 s. Clips of 4.0 s or more are
cropped identically under both modes, so their rows are already correct and are
left untouched.

WHY
    The extractor used to append digital silence to short clips. 71% of
    ASVspoof19 and 61% of In-the-Wild clips are under 4 s, against 3.8% of
    IndicSynth -- so a silent tail marks the corpus, not real vs fake. The live
    server and diagnose_pairs.py already repeat-pad. This brings the cached
    embeddings in line without re-extracting the long clips.

WHAT IT NEEDS
    The same source you extracted from, so every row's audio can be found:
      --audio-dir <folder>                     (any flat folder root)
      --data-root <asvspoof19_la> --split train|dev|eval
      --itw-root <folder with meta.csv> --split itw
    Layers and pooling are read from the root's meta.json -- nothing to match
    by hand.

SAFE TO STOP
    Each shard is rewritten atomically and logged in repad_progress.txt; a rerun
    skips finished shards. meta.json says pad=repeat only once every shard is
    done, and train.py refuses a root that is half-converted or whose pad mode
    differs from the other roots.

USAGE (Windows cmd, repo root)
    python repad_short.py --emb-root D:\\SONIX\\outputs\\embeddings_v2_mlaad --split train ^
        --audio-dir D:\\SONIX\\data\\mlaad_indic --batch 8 --workers 6
    python repad_short.py --emb-root D:\\SONIX\\outputs\\embeddings_v2 --split eval ^
        --data-root F:\\asvspoof19_la --batch 8 --workers 6
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import extract_embeddings as ee  # noqa: E402


def src_len(path):
    """Length in 16 kHz samples. Header first (cheap); full decode if needed."""
    try:
        import soundfile as sf
        i = sf.info(str(path))
        return int(round(i.frames * ee.TARGET_SR / i.samplerate))
    except Exception:
        return len(ee.load_audio_raw(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="zero-pad -> repeat-pad, short clips only")
    ap.add_argument("--emb-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--data-root", default="data/asvspoof19_la")
    ap.add_argument("--itw-root", default="data/in_the_wild")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="count short rows per shard, touch nothing")
    args = ap.parse_args()

    d = Path(args.emb_root) / args.split
    mp = d / "meta.json"
    if not mp.exists():
        sys.exit(f"FATAL: {mp} not found. Only finished v2 roots can be repadded.")
    meta = json.loads(mp.read_text())
    if meta.get("pad") == "repeat":
        print(f"{d}: already pad=repeat. Nothing to do.")
        return 0

    man = ee.build_manifest(args.split, SimpleNamespace(
        audio_dir=args.audio_dir, data_root=args.data_root, itw_root=args.itw_root))
    where = {}
    for name, path, _ in man:
        where[name] = path
        where[Path(path).name] = path
        where[Path(path).stem] = path

    shards = sorted(s for s in d.glob("shard_*.npy") if ".labels" not in s.name)
    if not shards:
        sys.exit(f"FATAL: no shards in {d}")
    prog_p = d / "repad_progress.txt"
    done = set(prog_p.read_text().split()) if prog_p.exists() else set()

    # locate every row first: a row with no audio would stay zero-padded, and a
    # half-converted root is worse than an unconverted one
    plan, missing = [], []
    for s in shards:
        names = [ln.strip() for ln in Path(str(s)[:-4] + ".files.txt")
                 .read_text(encoding="utf-8").splitlines() if ln.strip()]
        paths = []
        for n in names:
            p = where.get(n) or where.get(Path(n).stem)
            if p is None:
                missing.append(n)
            paths.append(p)
        plan.append((s, names, paths))
    if missing:
        sys.exit(f"FATAL: {len(missing)} rows have no audio under the source you "
                 f"gave (e.g. {missing[:3]}). Pass the same --audio-dir / "
                 f"--data-root / --itw-root the root was extracted from.")

    pool = ThreadPoolExecutor(max(1, args.workers))
    frontend = embed = None
    total_short = total_rows = 0
    print(f"{d}: {len(shards)} shards, blocks={meta['layers']} pool={meta['pool']}, "
          f"{len(done)} already converted")

    for k, (s, names, paths) in enumerate(plan, 1):
        if s.name in done:
            continue
        lens = list(pool.map(src_len, paths))
        short = [i for i, n in enumerate(lens) if n < ee.TARGET_LEN]
        total_rows += len(names)
        total_short += len(short)
        if args.dry_run:
            continue
        if short:
            if frontend is None:
                from extract_embeddings_v2 import load_frontend_v2, make_embed_batch
                layers = [b for b in meta["layers"] if b != "ln"]
                final_ln = "ln" in meta["layers"]
                device = args.device or ee._auto_device()
                frontend = load_frontend_v2(meta.get("model", "facebook/wav2vec2-xls-r-300m"),
                                            device)
                embed = make_embed_batch(layers, meta["pool"], final_ln)
            X = np.load(s)
            if X.shape[1] != meta["total_dim"]:
                sys.exit(f"FATAL: {s.name} is {X.shape[1]}-dim, meta says "
                         f"{meta['total_dim']}")
            wavs = list(pool.map(
                lambda i: ee.fit_window(ee.load_audio_raw(paths[i]), "repeat"), short))
            new = []
            for b in range(0, len(wavs), args.batch):
                new.extend(embed(frontend, wavs[b:b + args.batch]))
            new = np.asarray(new, dtype=X.dtype)
            if not np.isfinite(new.astype(np.float32)).all():
                sys.exit(f"FATAL: non-finite embeddings in {s.name}; nothing "
                         f"written for this shard. Try --device cpu to confirm.")
            X[short] = new
            ee._atomic_np_save(s, X)
        with open(prog_p, "a", encoding="utf-8") as fh:
            fh.write(s.name + "\n")
        if k % 20 == 0 or k == len(plan):
            print(f"  {k}/{len(plan)} shards  short rows so far: {total_short}",
                  flush=True)

    pct = 100 * total_short / max(total_rows, 1)
    if args.dry_run:
        print(f"DRY RUN: {total_short} of {total_rows} rows ({pct:.1f}%) are "
              f"short and would be re-embedded.")
        return 0
    meta["pad"] = "repeat"
    meta["repadded_rows"] = int(meta.get("repadded_rows", 0)) + total_short
    mp.write_text(json.dumps(meta, indent=2))
    (d / "pad_mode.txt").write_text("repeat\n")
    print(f"DONE {d}: re-embedded {total_short} short rows ({pct:.1f}% of the "
          f"rows converted this run); meta.json now says pad=repeat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
