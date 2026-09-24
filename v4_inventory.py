#!/usr/bin/env python3
"""
v4_inventory.py  --  SONIX / SIH26104

Read-only stock-take of a drive or folder: what audio, what embeddings, what
format. Run it on the SSD and on each machine BEFORE anything is copied,
extracted or deleted, so nobody works from memory.

    python v4_inventory.py F:\\
    python v4_inventory.py E:\\ --out ssd_inventory.md
    python v4_inventory.py D:\\sonix_data --max-depth 6

It never writes anywhere except the --out report.

WHAT IT REPORTS, per folder that directly holds data
  * audio files by extension, total size, tiny/zero-byte files
  * embedding shards: row count, and whether meta.json says the v4 format
      (layers [5, 6, 24], pool meanstd, total_dim 6144) -- anything else is v1
      or unknown and cannot be co-trained
  * label counts from *.labels.npy (all-zero on a fake folder = unstamped)
  * archives (.zip / .tgz / .tar / .gz) with size, so half-finished downloads show
  * duplicate audio filename stems across folders -- train.py joins the manifest
    on the stem, so two different clips with one stem silently collide
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
ARCHIVE_EXT = (".zip", ".tgz", ".tar", ".gz", ".7z", ".rar")
SKIP_DIRS = {"$recycle.bin", "system volume information", ".git", "node_modules",
             ".venv", "__pycache__", ".cache", "appdata"}
V4 = {"layers": [5, 6, 24], "pool": "meanstd", "total_dim": 6144}

# path keyword -> dataset name (first match wins, most specific first)
KEYWORDS = (
    ("indicvoices", "IndicVoices"), ("indic_voices", "IndicVoices"),
    ("indicsynth", "IndicSynth"), ("mlaad", "MLAAD"),
    ("librisevoc", "LibriSeVoc"), ("musan", "MUSAN"),
    ("rirs", "RIRS_NOISES"), ("rir_", "RIRS_NOISES"),
    ("common_voice", "Common Voice"), ("commonvoice", "Common Voice"),
    ("cv-corpus", "Common Voice"), ("in_the_wild", "In-the-Wild"),
    ("release_in_the_wild", "In-the-Wild"), ("itw", "In-the-Wild"),
    ("asvspoof5", "ASVspoof 5"), ("asvspoof2021", "ASVspoof 2021 DF"),
    ("df21", "ASVspoof 2021 DF"), ("asvspoof", "ASVspoof 2019 LA"),
    ("mms_tts", "MMS-TTS"), ("indic_spoof", "MMS-TTS"),
    ("sonix_real", "sonix_real"), ("deep-voice", "DEEP-VOICE"),
    ("deep_voice", "DEEP-VOICE"), ("wavefake", "WaveFake"),
    ("xtts", "Zero-shot clones"), ("openvoice", "Zero-shot clones"),
    ("zero_shot", "Zero-shot clones"), ("zeroshot", "Zero-shot clones"),
)


def guess_dataset(path: str) -> str:
    low = path.lower().replace("\\", "/")
    for kw, name in KEYWORDS:
        if kw in low:
            return name
    return "?"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def read_meta(d: Path):
    p = d / "meta.json"
    if not p.exists():
        return None, "no meta.json (v1 or unknown)"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return None, f"meta.json unreadable: {e}"
    ok = (list(m.get("layers", [])) == V4["layers"] and m.get("pool") == V4["pool"]
          and int(m.get("total_dim", -1)) == V4["total_dim"])
    desc = f"layers={m.get('layers')} pool={m.get('pool')} dim={m.get('total_dim')}"
    return ok, ("v4 OK  " if ok else "NOT v4 ") + desc


def count_rows_and_labels(d: Path, shard_files):
    rows = 0
    for f in shard_files:
        if f.endswith(".files.txt"):
            try:
                with open(d / f, encoding="utf-8", errors="replace") as fh:
                    rows += sum(1 for line in fh if line.strip())
            except OSError:
                pass
    labels = Counter()
    try:
        import numpy as np
        for f in shard_files:
            if f.endswith(".labels.npy"):
                try:
                    arr = np.load(d / f, mmap_mode="r")
                    for v, c in zip(*np.unique(np.asarray(arr), return_counts=True)):
                        labels[int(v)] += int(c)
                except Exception:  # noqa: BLE001
                    labels["unreadable"] += 1
    except ImportError:
        labels["numpy missing"] = 1
    return rows, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="drive or folder to scan, e.g. F:\\")
    ap.add_argument("--out", default="", help="markdown report path "
                    "(default: v4_inventory_<name>.md in the current folder)")
    ap.add_argument("--max-depth", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"FATAL: {root} does not exist")
    root_depth = len(root.resolve().parts)

    data_dirs = []            # (path, dataset, audio Counter, bytes, tiny, shards)
    archives = []             # (path, size)
    stems = defaultdict(list)
    n_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root):
        n_dirs += 1
        depth = len(Path(dirpath).resolve().parts) - root_depth
        dirnames[:] = [d for d in dirnames
                       if d.lower() not in SKIP_DIRS and not d.startswith(".")
                       and depth < args.max_depth]
        audio, size, tiny = Counter(), 0, 0
        shard_files = []
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(dirpath, f)
            if ext in AUDIO_EXT:
                audio[ext] += 1
                try:
                    s = os.path.getsize(full)
                except OSError:
                    s = 0
                size += s
                if s < 1024:
                    tiny += 1
                stems[os.path.splitext(f)[0].lower()].append(dirpath)
            elif f.lower().endswith(ARCHIVE_EXT):
                try:
                    archives.append((full, os.path.getsize(full)))
                except OSError:
                    archives.append((full, -1))
            elif f.startswith("shard_") and (f.endswith(".npy") or f.endswith(".files.txt")):
                shard_files.append(f)
        if audio or shard_files:
            data_dirs.append((dirpath, guess_dataset(dirpath), audio, size, tiny,
                              sorted(shard_files)))
        if n_dirs % 2000 == 0:
            print(f"  ... scanned {n_dirs} folders", file=sys.stderr)

    out_lines = [f"# v4 inventory — {root}", ""]

    # ---- audio
    audio_dirs = [d for d in data_dirs if d[2]]
    out_lines += ["## Audio", "",
                  "| dataset (guess) | folder | files | size | tiny <1KB |",
                  "|---|---|---|---|---|"]
    per_ds_audio = Counter()
    for path, ds, audio, size, tiny, _ in sorted(audio_dirs, key=lambda x: (x[1], x[0])):
        n = sum(audio.values())
        per_ds_audio[ds] += n
        exts = " ".join(f"{k}:{v}" for k, v in audio.most_common())
        out_lines.append(f"| {ds} | `{path}` | {n} ({exts}) | {human(size)} | "
                         f"{tiny if tiny else ''} |")
    if not audio_dirs:
        out_lines.append("| — | no audio found | | | |")

    # ---- embeddings
    emb_dirs = [d for d in data_dirs if d[5]]
    out_lines += ["", "## Embeddings", "",
                  "| dataset (guess) | folder | shards | rows | labels | format |",
                  "|---|---|---|---|---|---|"]
    for path, ds, _, _, _, shard_files in sorted(emb_dirs, key=lambda x: (x[1], x[0])):
        d = Path(path)
        n_shards = sum(1 for f in shard_files
                       if f.endswith(".npy") and not f.endswith(".labels.npy"))
        rows, labels = count_rows_and_labels(d, shard_files)
        _, fmt = read_meta(d)
        lab = " ".join(f"{k}:{v}" for k, v in sorted(labels.items(), key=str))
        out_lines.append(f"| {ds} | `{path}` | {n_shards} | {rows} | {lab} | {fmt} |")
    if not emb_dirs:
        out_lines.append("| — | no embedding shards found | | | | |")

    # ---- archives
    out_lines += ["", "## Archives (check against expected download size)", ""]
    if archives:
        for path, s in sorted(archives):
            out_lines.append(f"- `{path}` — {human(s) if s >= 0 else 'unreadable'}")
    else:
        out_lines.append("- none")

    # ---- totals
    out_lines += ["", "## Audio totals by dataset", ""]
    for ds, n in per_ds_audio.most_common():
        out_lines.append(f"- {ds}: {n}")

    # ---- stem collisions
    dup = {s: dirs for s, dirs in stems.items() if len(dirs) > 1}
    out_lines += ["", "## Duplicate filename stems across folders", ""]
    if dup:
        out_lines.append(f"{len(dup)} stems appear in more than one folder. "
                         "train.py joins metadata on the stem, so these must be "
                         "renamed with a corpus/generator prefix before extraction.")
        out_lines.append("")
        for s, dirs in list(dup.items())[:15]:
            out_lines.append(f"- `{s}` ×{len(dirs)}: " + ", ".join(
                f"`{x}`" for x in sorted(set(dirs))[:3]))
    else:
        out_lines.append("- none")

    report = "\n".join(out_lines) + "\n"
    name = (root.resolve().name or str(root.resolve().drive).rstrip(":\\") or "root")
    out = Path(args.out or f"v4_inventory_{name}.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out.resolve()}  ({n_dirs} folders scanned)")


if __name__ == "__main__":
    main()
