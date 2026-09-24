#!/usr/bin/env python3
"""Common Voice Indic -- speaker partition first, audio second.

WHY THE PARTITION COMES FIRST
Common Voice bonafide goes into v4 TRAINING, and Akshat's training clones use
Common Voice speakers as reference audio. If his references come from speakers
whose real clips land in the eval half, the same speaker straddles the split and
the eval number is inflated (V4_BUILD_BRIEF §2.3, recording-level holdout).

He is blocked on a list of client_ids, not on audio. The TSVs carry those and
are a few MB, so `--speakers-only` finishes in minutes while the ~15k-clip audio
pull runs afterwards.

The split is a hash of client_id, so it is identical on every machine and stable
across reruns -- re-running this must never silently move a speaker.

    pip install -U huggingface_hub
    huggingface-cli login          # and accept the terms on the dataset page

    python prep_common_voice.py --speakers-only
    python prep_common_voice.py --langs hi,bn,ta,mr,pa,ur,te --per-lang 2200 \
        --out data/common_voice_indic
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = "mozilla-foundation/common_voice_17_0"
DEFAULT_LANGS = "hi,bn,ta,mr,pa,ur"          # te is NOT in here by default; see --langs
TRAIN_PCT = 80                                # speakers, not clips


def speaker_split(client_id: str, train_pct: int = TRAIN_PCT) -> str:
    """Deterministic, machine-independent, stable across reruns.

    Python's hash() is salted per process and would reshuffle the split on every
    run, so use sha256 explicitly.
    """
    h = int(hashlib.sha256(client_id.encode()).hexdigest()[:8], 16)
    return "train" if h % 100 < train_pct else "eval"


def repo_tsvs(langs):
    """Locate the per-language metadata TSVs without downloading audio.

    The repo is gated, so the layout is not verified here -- discover it and
    fail with the real listing rather than hard-coding a path that may differ.
    """
    from huggingface_hub import list_repo_files
    files = list_repo_files(REPO, repo_type="dataset")
    found = defaultdict(list)
    for f in files:
        if not f.endswith(".tsv"):
            continue
        parts = f.split("/")
        for lang in langs:
            if lang in parts:
                found[lang].append(f)
    missing = [l for l in langs if not found.get(l)]
    if missing:
        sample = "\n  ".join(sorted(x for x in files if x.endswith(".tsv"))[:20])
        sys.exit(f"FATAL: no .tsv found for {missing}.\n"
                 f"  Check the language codes against the repo. First TSVs seen:\n  {sample}")
    return dict(found)


def client_ids(tsv_paths):
    """-> {lang: {client_id: clip_count}}  from downloaded TSVs only."""
    from huggingface_hub import hf_hub_download
    out = {}
    for lang, files in tsv_paths.items():
        counts = Counter()
        for f in sorted(files):
            local = hf_hub_download(REPO, f, repo_type="dataset")
            with open(local, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                if "client_id" not in (reader.fieldnames or []):
                    print(f"  ! {f}: no client_id column, skipped")
                    continue
                for row in reader:
                    cid = (row.get("client_id") or "").strip()
                    if cid:
                        counts[cid] += 1
        out[lang] = counts
        print(f"  {lang}: {len(counts)} speakers, {sum(counts.values())} clips")
    return out


def check_licence():
    """Verify the CC0 claim against the release itself. DOWNLOADS.md says CC0;
    a paper needs the field, not the hearsay."""
    from huggingface_hub import dataset_info
    try:
        info = dataset_info(REPO)
        lic = getattr(info, "card_data", None)
        lic = (lic or {}).get("license") if lic else None
        print(f"\nlicence field on {REPO}: {lic!r}")
        if not lic or "cc0" not in str(lic).lower():
            print("  ! NOT the CC0 string DOWNLOADS.md claims. Read the dataset card\n"
                  "    and the Common Voice terms before this appears in the paper.")
    except Exception as e:
        print(f"\n! could not read licence field: {e}\n  Check it by hand.")


def write_partition(per_lang, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "eval": []}
    for lang, counts in per_lang.items():
        for cid in counts:
            buckets[speaker_split(cid)].append(f"{cid}\t{lang}\t{counts[cid]}")
    paths = {}
    for k, rows in buckets.items():
        p = outdir / f"cv_{k}_speakers.txt"
        p.write_text("\n".join(sorted(rows)) + "\n", encoding="utf-8")
        paths[k] = p
        print(f"wrote {p}  ({len(rows)} speakers)")

    overlap = ({r.split('\t')[0] for r in buckets['train']}
               & {r.split('\t')[0] for r in buckets['eval']})
    if overlap:   # a client_id reused across languages would land in both files
        sys.exit(f"FATAL: {len(overlap)} client_id(s) in both halves -- "
                 f"the hash split is not being applied consistently.")
    return paths


def wanted_clips(lang, tsv_files, half, per_lang, per_speaker):
    """-> {clip filename: client_id} for one language, capped.

    Cap PER SPEAKER, not just per language. Common Voice is heavily skewed --
    a handful of prolific contributors can supply most of a config, and an
    uncapped pull would hand the model a few voices to memorise instead of the
    channel and speaker diversity this download exists to provide.
    """
    from huggingface_hub import hf_hub_download
    rows = []
    for f in sorted(tsv_files):
        local = hf_hub_download(REPO, f, repo_type="dataset")
        with open(local, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="	"):
                cid, name = (row.get("client_id") or "").strip(), (row.get("path") or "").strip()
                if cid and name and speaker_split(cid) == half:
                    rows.append((name, cid))
    rows.sort()                       # deterministic selection across machines
    taken, per_cid = {}, Counter()
    for name, cid in rows:
        if len(taken) >= per_lang:
            break
        if per_cid[cid] >= per_speaker:
            continue
        per_cid[cid] += 1
        taken[name] = cid
    print(f"  {lang}: {len(taken)} clips from {len(per_cid)} {half}-half speakers")
    return taken


def fetch_audio(langs, per_lang_speakers, half, per_lang, per_speaker, out: Path):
    """Pull only the clips selected above out of the language tars.

    The tar is downloaded whole -- it is one file and the hub cannot serve part
    of it -- but only selected members are written to disk.
    """
    import tarfile
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(REPO, repo_type="dataset")
    tsvs = repo_tsvs(langs)
    for lang in langs:
        tars = sorted(f for f in files
                      if f.endswith(".tar") and lang in f.split("/"))
        if not tars:
            print(f"  ! {lang}: no .tar found in the repo listing, skipped")
            continue
        want = wanted_clips(lang, tsvs[lang], half, per_lang, per_speaker)
        dest = out / half / lang
        dest.mkdir(parents=True, exist_ok=True)
        written = 0
        for t in tars:
            if written >= len(want):
                break
            local = hf_hub_download(REPO, t, repo_type="dataset")
            with tarfile.open(local) as tf:
                for m in tf:
                    base = Path(m.name).name
                    if not m.isfile() or base not in want:
                        continue
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    (dest / base).write_bytes(src.read())
                    written += 1
        print(f"  {lang}: wrote {written}/{len(want)} to {dest}")
        if written < len(want):
            print(f"    ! {len(want) - written} selected clip(s) not found in any tar")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default=DEFAULT_LANGS,
                    help=f"comma-separated config names (default {DEFAULT_LANGS}). "
                         f"Add 'te' if Telugu volunteers are being recorded -- "
                         f"otherwise Telugu has no bonafide coverage at all.")
    ap.add_argument("--speakers-only", action="store_true",
                    help="fetch TSVs and write the partition, no audio. "
                         "This is what unblocks the training-clone generation.")
    ap.add_argument("--partition-out", default="data",
                    help="where cv_train_speakers.txt / cv_eval_speakers.txt go")
    ap.add_argument("--per-lang", type=int, default=2500, help="audio clips per language")
    ap.add_argument("--half", default="train", choices=["train", "eval"],
                    help="which speaker half to pull audio for. 'train' feeds v4 "
                         "training; 'eval' is the held-out bonafide set and must "
                         "never be passed to train.py.")
    ap.add_argument("--per-speaker", type=int, default=12,
                    help="max clips per contributor -- Common Voice is skewed and "
                         "an uncapped pull is a few voices, not diversity")
    ap.add_argument("--out", default="data/common_voice_indic")
    args = ap.parse_args()

    langs = [l.strip().lower() for l in args.langs.split(",") if l.strip()]
    if "te" not in langs:
        print("! Telugu ('te') not in --langs. If Telugu volunteers are being\n"
              "  recorded, they will have no bonafide coverage in training.\n")

    check_licence()
    print(f"\nlocating TSVs for {langs} ...")
    per_lang = client_ids(repo_tsvs(langs))
    paths = write_partition(per_lang, Path(args.partition_out))

    print(f"\nSEND {paths['train']} TO AKSHAT BEFORE HE GENERATES ANYTHING.")
    if args.speakers_only:
        print("(--speakers-only: no audio fetched. Rerun without it for clips.)")
        return

    fetch_audio(langs, per_lang, args.half, args.per_lang,
                args.per_speaker, Path(args.out))


if __name__ == "__main__":
    main()
