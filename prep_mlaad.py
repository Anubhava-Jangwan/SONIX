"""Fetch the Indic slice of MLAAD -- real synthetic Indian speech, many voices.

WHY THIS AND NOT make_indic_spoof.py
make_indic_spoof.py gives you ONE MMS-TTS voice per language. A head trained on
that learns "this particular synthesiser", not "synthesis". MLAAD is 140 TTS
models across 51 languages, and its Indic slice is roughly 45 hours:

    Hindi 15.1h · Bangla 14.1h · Tamil 5.0h · Kannada 4.2h
    Marathi 3.0h · Urdu 2.1h · Malayalam 1.8h

Paired with IndicVoices on the bonafide side, that puts the SAME languages on
both sides of the label. That is the whole point: while Hindi only ever appears
as bonafide, "Hindi" stays a shortcut to "real" and a cloned Hindi voice walks
straight through -- which is exactly what head_full_ho does today.

MLAAD is CC BY-NC 4.0: research and this prototype are fine, shipping it
commercially is not. Worth knowing before it goes in a product pitch.

    pip install -U huggingface_hub

    python prep_mlaad.py --list                       # what languages exist
    python prep_mlaad.py --langs hi,bn,ta,mr,kn,ml,ur --out data/mlaad_indic
    python prep_mlaad.py --langs hi,bn,ta --flatten-only --out data/mlaad_indic

The full repo is 183 GB. This downloads only the language folders you name.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = "mueller91/MLAAD"
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")

# ISO-639-1 and -3 spellings both appear in the wild; match either.
ALIASES = {
    "hi": ("hi", "hin", "hindi"),
    "bn": ("bn", "ben", "bangla", "bengali"),
    "ta": ("ta", "tam", "tamil"),
    "mr": ("mr", "mar", "marathi"),
    "kn": ("kn", "kan", "kannada"),
    "ml": ("ml", "mal", "malayalam"),
    "ur": ("ur", "urd", "urdu"),
    "te": ("te", "tel", "telugu"),
    "gu": ("gu", "guj", "gujarati"),
    "pa": ("pa", "pan", "punjabi"),
}


def repo_languages():
    from huggingface_hub import list_repo_files
    files = list_repo_files(REPO, repo_type="dataset")
    langs = {}
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == "fake":
            langs.setdefault(parts[1], set()).add(parts[2])
    return langs, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the language folders MLAAD actually has, then exit")
    ap.add_argument("--langs", default="hi,bn,ta,mr,kn,ml,ur",
                    help="comma-separated language codes to fetch")
    ap.add_argument("--out", default="data/mlaad_indic",
                    help="flat folder of spoof wavs for extract_embeddings")
    ap.add_argument("--cache", default="data/mlaad_raw",
                    help="where the downloaded tree lands")
    ap.add_argument("--per-model", type=int, default=0,
                    help="cap clips taken per TTS model (0 = all). Use this to "
                         "keep one prolific model from dominating the class.")
    ap.add_argument("--flatten-only", action="store_true",
                    help="skip the download, just re-flatten what is cached")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install -U huggingface_hub")

    if args.list:
        langs, _ = repo_languages()
        print(f"{len(langs)} language folders in {REPO}:\n")
        for k in sorted(langs):
            print(f"  {k:<12} {len(langs[k])} TTS model folders")
        print("\nPick the Indic ones and pass them to --langs.")
        return 0

    wanted = [w.strip().lower() for w in args.langs.split(",") if w.strip()]
    cache = Path(args.cache)

    if not args.flatten_only:
        langs, _ = repo_languages()
        have = set(langs)
        patterns, missing = [], []
        for w in wanted:
            names = [n for n in ALIASES.get(w, (w,)) if n in have]
            if names:
                patterns += [f"fake/{n}/**" for n in names]
            else:
                missing.append(w)
        if missing:
            print(f"  ! not present in the repo, skipping: {', '.join(missing)}")
            print(f"    run --list to see the real folder names")
        if not patterns:
            sys.exit("none of the requested languages exist in the repo")

        print(f"downloading {len(patterns)} language folders into {cache} ...")
        print("  (this is a subset of a 183 GB repo -- it resumes if interrupted)")
        snapshot_download(REPO, repo_type="dataset", local_dir=str(cache),
                          allow_patterns=patterns + ["*.csv", "*.txt", "*.md"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # extract_embeddings.py --audio-dir globs *.wav NON-recursively, so the
    # per-model tree has to be flattened into one folder. Keep the language and
    # model in the filename so the set stays auditable afterwards.
    n, per_model = 0, {}
    root = cache / "fake"
    if not root.is_dir():
        sys.exit(f"no {root} -- download first (drop --flatten-only)")

    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in AUDIO_EXT:
            continue
        rel = f.relative_to(root).parts
        lang = rel[0] if len(rel) > 1 else "unk"
        model = rel[1] if len(rel) > 2 else "unk"
        key = (lang, model)
        if args.per_model and per_model.get(key, 0) >= args.per_model:
            continue
        per_model[key] = per_model.get(key, 0) + 1
        shutil.copy2(f, out / f"mlaad_{lang}_{model}_{per_model[key]:05d}{f.suffix.lower()}")
        n += 1

    print(f"\n{n} spoof clips -> {out}")
    print(f"{len(per_model)} (language, model) pairs:")
    for (lang, model), c in sorted(per_model.items())[:40]:
        print(f"  {lang:<6} {model:<40} {c}")
    if len(per_model) > 40:
        print(f"  ... and {len(per_model) - 40} more")

    print()
    print("NEXT -- extract, then stamp as SPOOF (label 1). Skipping the stamp")
    print("means every one of these trains as genuine, which is worse than not")
    print("having them at all:")
    print()
    print(f"  python src\\extract_embeddings.py --split train --audio-dir {out} \\")
    print(f"      --out outputs\\embeddings_mlaad_indic --batch 8")
    print(f"  python stamp_labels.py --emb-dir outputs\\embeddings_mlaad_indic\\train --label 1")
    print(f"  python stamp_labels.py --emb-dir outputs\\embeddings_mlaad_indic\\train --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
