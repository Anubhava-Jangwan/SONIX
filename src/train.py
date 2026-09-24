#!/usr/bin/env python3
"""
train.py  --  SONIX / SIH26104        v4

Trains the head on cached SSL embeddings. The front-end is frozen and its
outputs are on disk, so this trains in MINUTES. If it is taking hours you are
re-extracting features somewhere -- stop and find it.

WHAT CHANGED IN v4  (docs/V4_BUILD_BRIEF.md section 6, in that order)

  1. GROUP-BALANCED SAMPLING replaces the global pos_weight.
     Groups are {language} x {generator_family} x {label}; each row is weighted
     N_total / (n_groups * n_g).

     Why: the v3 split gives P(spoof | English) = 89.8% against
     P(spoof | Indic) = 48.7% -- a 41-point language prior. pos_weight balances
     only the global positive/negative ratio, so it leaves that prior fully
     intact, and docs/PROBE_RESULTS.md shows language is 99.33% linearly
     decodable from the vector. "It is English, so it is probably spoof" is
     therefore both available and rewarded.

     The consequence is measured. On the 21 matched Indian-celebrity pairs in
     outputs/diagnose_pairs.csv, the two heads trained on IndicVoices score
     AUC 0.3545 (full_ho) and 0.5091 (v3) -- chance or worse -- while
     robust_v2, which never saw IndicVoices, scores 0.7636. n=21, so the
     ordering is solid and the exact values are not.

  2. MODEL SELECTION ON WORST-GROUP EER, not mean. Both are reported.
     Selecting on the mean is how head_v3 shipped with its Indic-fake cell
     collapsed to zero contribution and the metric never noticing.

  3. MULTITASK HEAD -- a second output naming the generator family.
     Vocoder artifacts are language-independent, so a model forced to name the
     generator cannot answer "it is Hindi". --aux-weight controls it; 0
     disables the branch entirely.

  Plus OC-Softmax behind --loss, so BCE and one-class can be ablated.

HARD RULES ENFORCED HERE  (brief section 2 -- violating any invalidates the run)
  * outputs/embeddings_indicvoices_ho can never enter training. Every resolved
    root is checked, because the documented failure is a shell glob on
    outputs/embeddings* swallowing it.
  * Recording-level holdout, never row-level. Dev is carved on recording_id so
    <id>_chunk_N siblings cannot straddle the split.
  * Never fabricate a number. Nothing here estimates or fills in a group label;
    a row whose group is unknown stops the run.

INPUT
    <root>/train/shard_*.npy + .labels.npy + .files.txt   [+ meta.json for v2]
    a manifest CSV carrying the group metadata (see --manifest)

OUTPUT
    outputs/models/head_v4.pt -- state_dict, standardiser, group table,
    per-group dev EER, and full provenance of what it was trained on.

USAGE
    python src/train.py --emb-root outputs/embeddings_v2 \\
        --manifest outputs/manifest_v4.csv --out outputs/models/head_v4.pt

    # v4.0, before the S1 manifest exists: one root per corpus / per vocoder
    python src/train.py --derive-groups --emb-root outputs/embeddings_v2 \\
        --extra-emb-root <...>_indicvoices_tr --extra-emb-root <...>_indicsynth \\
        --extra-emb-root <...>_mlaad --extra-emb-root <...>_librisevoc_gt \\
        --extra-emb-root <...>_librisevoc_diffwave   (one per vocoder) \\
        --lang-table docs/v4_splits/chunks.csv \\
        --durations <duration csvs> --min-dur 4.0

    LibriSeVoc: files are NOT renamed. Every vocoder names an utterance
    <utt>_gen, so the vocoder comes from the root name (..._librisevoc_<voc>)
    and gt/fakes of one speaker share a dev-carve id.

    # ablate the pieces
    python src/train.py ... --loss bce --aux-weight 0.0     # v3 behaviour + groups
    python src/train.py ... --loss ocsoftmax --aux-weight 0.3
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# The one root that must never be trained on. Brief section 2 rule 1, and
# section 5 lists it as held out. 52,402 - 5,050 = 47,352 is the arithmetic
# that tells you whether it leaked in.
FORBIDDEN = "embeddings_indicvoices_ho"


# ---------------------------------------------------------------------------
# EER -- local cross-check copy. src/metrics.py:31 calculate_eer() is the
# source of truth for any REPORTED number; this exists so training can
# early-stop without importing the plotting stack, and so the two can be
# compared. labels: 1 = spoof = positive class.
# ---------------------------------------------------------------------------
def eer(labels, scores) -> float:
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan")          # a one-class cell has no EER; caller reports it
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ===========================================================================
# Loading
# ===========================================================================
def assert_not_forbidden(root: str) -> None:
    """Stop the run if a root is, or contains, the held-out IndicVoices set.

    Checked on the RESOLVED path rather than the string passed in, because the
    documented way this leaks is a shell glob -- `--emb-root outputs/embeddings*`
    expands to include it and nothing in the old code noticed.
    """
    p = Path(root).resolve()
    parts = [q.name for q in [p, *p.parents]]
    if any(FORBIDDEN in part for part in parts):
        sys.exit(
            f"FATAL: {root} resolves to {p}, which is the held-out IndicVoices\n"
            f"       set ({FORBIDDEN}). Brief section 2 rule 1: it never enters\n"
            f"       training, at any stage. If you reached here through a glob\n"
            f"       like outputs/embeddings*, that is exactly the documented\n"
            f"       failure -- name the roots explicitly.")


def read_meta(d: Path, dim: int):
    """Return the meta.json sidecar for an embedding root, or a legacy stand-in.

    extract_embeddings_v2.py writes meta.json as the LAST step of main(), after
    every shard. So a root with shards and no meta.json is an extraction that
    is still running or was killed -- its layer layout is unknown and slicing
    it would be a guess. That is fatal.

    A 1024-dim root is the v1 extractor, which predates the sidecar. Those are
    known to be last_hidden_state mean-pooled, so they get a synthetic meta
    rather than an error.
    """
    mp = d / "meta.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        want = len(m["layers"]) * m["dim_per_layer"]
        if want != dim:
            sys.exit(f"FATAL: {mp} declares {len(m['layers'])} blocks x "
                     f"{m['dim_per_layer']} = {want} dims but the shards are "
                     f"{dim}. The sidecar does not match the data.")
        return m
    if dim == 1024:
        return {"model": "facebook/wav2vec2-xls-r-300m", "layers": ["ln"],
                "pool": "mean", "dim_per_layer": 1024, "total_dim": 1024,
                "_synthetic": "legacy v1 root: last_hidden_state, mean-pooled"}
    sys.exit(
        f"FATAL: {d} has {dim}-dim shards and no meta.json.\n"
        f"       extract_embeddings_v2.py writes that sidecar only after the\n"
        f"       LAST shard, so this is an extraction still running or killed\n"
        f"       part-way. Without it the layer layout is unknown and --layer\n"
        f"       would be a guess. Let the extraction finish, or delete the\n"
        f"       root and re-run it.")


def load_root(root: str, split: str, layer=None):
    """One embedding root/split -> (X float32, y int64, filenames, meta).

    Keeps the shard/sidecar contract of the v1 loader: shard k covers manifest
    rows [k*size, (k+1)*size), a short shard is fine, and the three sidecars
    stay row-aligned so labels never drift out of sync with embeddings.
    """
    assert_not_forbidden(root)
    d = Path(root) / split
    shards = [s for s in sorted(d.glob("shard_*.npy")) if ".labels" not in s.name]
    if not shards:
        sys.exit(f"FATAL: no shards in {d}. Extract this split first.")

    Xs, ys, files = [], [], []
    for s in shards:
        lab = Path(str(s)[:-4] + ".labels.npy")
        nam = Path(str(s)[:-4] + ".files.txt")
        for p in (lab, nam):
            if not p.exists():
                sys.exit(f"FATAL: {s.name} has no matching {p.name}. The three "
                         f"sidecars must stay together; re-extract this shard.")
        emb, y = np.load(s), np.load(lab)
        names = [ln.strip() for ln in nam.read_text().splitlines() if ln.strip()]
        if not (len(emb) == len(y) == len(names)):
            sys.exit(f"FATAL: {s.name} rows {len(emb)} / labels {len(y)} / names "
                     f"{len(names)} disagree. Corrupt shard; delete and re-extract.")
        if len(emb) == 0:
            continue                  # a batch that failed at extraction time
        Xs.append(emb)
        ys.append(y)
        files += names

    if not Xs:
        sys.exit(f"FATAL: every shard in {d} is empty.")
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0).astype(np.int64)
    meta = read_meta(d, X.shape[1])

    if layer is not None:
        labels = meta["layers"]
        key = layer if layer in labels else _coerce_layer(layer, labels, d)
        i, P = labels.index(key), meta["dim_per_layer"]
        X = X[:, i * P:(i + 1) * P]
        meta = {**meta, "layers": [key], "total_dim": P, "_sliced_from": labels}

    X = X.astype(np.float32)
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(1).sum())
        sys.exit(f"FATAL: {d} has {n_bad} rows containing NaN/inf. This is the "
                 f"fp16 failure in PROJECT_STATE section 3 -- the extractor's "
                 f"self-check validates shape only. Re-extract this root.")

    print(f"  {root:44s} {split:5s} n={len(X):6d} dim={X.shape[1]:5d} "
          f"blocks={meta['layers']} pool={meta['pool']} "
          f"[spoof={int((y == 1).sum())} bona={int((y == 0).sum())}]")
    return X, y, files, meta


def _coerce_layer(layer, labels, d):
    """--layer comes off argparse as a string; block labels are ints plus "ln"."""
    try:
        v = int(layer)
    except (TypeError, ValueError):
        sys.exit(f"FATAL: --layer {layer!r} is not a block in {d}; "
                 f"that root has {labels}")
    if v not in labels:
        sys.exit(f"FATAL: --layer {layer!r} is not a block in {d}; "
                 f"that root has {labels}")
    return v


# ===========================================================================
# Group metadata -- {language} x {generator_family} x {label}
# ===========================================================================
MANIFEST_COLUMNS = ("path", "recording_id", "corpus", "language",
                    "generator_family", "channel", "label", "split")


def load_manifest(path: str):
    """S1 manifest -> {join_key: row}. Join key is the filename stem, which is
    what the shard .files.txt sidecars store.

    This is the preferred source and the only one that is not inference. The
    schema is brief section 1's, verbatim.

    STEM COLLISIONS. LibriSeVoc names every vocoder's copy of an utterance the
    same (<utt>_gen), so six roots share one stem. Keyed by stem alone, the last
    row read would silently win for all six. A manifest with repeated stems must
    therefore carry an extra `emb_root` column (the embedding root's folder
    name), and the join becomes (emb_root, stem). Repeated stems without it stop
    the run rather than join to the wrong row.
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"FATAL: --manifest {p} not found.")
    with open(p, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        missing = [c for c in MANIFEST_COLUMNS if c not in (rdr.fieldnames or [])]
        if missing:
            sys.exit(f"FATAL: {p} is missing column(s) {missing}.\n"
                     f"       S1 schema is: {', '.join(MANIFEST_COLUMNS)}")
        by_root = "emb_root" in (rdr.fieldnames or [])
        out, dup = {}, []
        for r in rdr:
            stem = Path(r["path"].replace("\\", "/")).stem
            key = (Path(r["emb_root"].replace("\\", "/")).name, stem) if by_root else stem
            if key in out:
                dup.append(key)
            r["language"] = norm_lang(r["language"], f"{p} row {stem}")
            out[key] = r
    if dup:
        hint = ("" if by_root else
                "\n       Add an `emb_root` column (the embedding root folder "
                "name) so the join\n       is (emb_root, stem) -- LibriSeVoc "
                "vocoders all share one stem per utterance.")
        sys.exit(f"FATAL: {len(dup)} repeated join key(s) in {p}, e.g. "
                 f"{dup[:3]}.{hint}")
    print(f"  manifest: {len(out)} rows from {p}"
          f"{'  (joined on emb_root + stem)' if by_root else ''}")
    return {"rows": out, "by_root": by_root}


# --- language ---------------------------------------------------------------
# One code per language across every corpus. Without this, IndicSynth says
# "bengali", MMS-TTS says "ben" and a manifest says "bn" -- three groups for one
# language, and eval_cells scores Bengali fakes against no Bengali bonafide.
LANG_ALIASES = {
    "en": ("en", "eng", "english"),
    "hi": ("hi", "hin", "hindi"),
    "bn": ("bn", "ben", "bengali", "bangla"),
    "ta": ("ta", "tam", "tamil"),
    "te": ("te", "tel", "telugu"),
    "mr": ("mr", "mar", "marathi"),
    "gu": ("gu", "guj", "gujarati"),
    "kn": ("kn", "kan", "kannada"),
    "ml": ("ml", "mal", "malayalam"),
    "pa": ("pa", "pan", "punjabi", "panjabi"),
    "or": ("or", "ory", "ori", "odia", "oriya"),
    "as": ("as", "asm", "assamese"),
    "ur": ("ur", "urd", "urdu"),
    "sa": ("sa", "san", "sanskrit"),
    "ne": ("ne", "nep", "npi", "nepali"),
    "mai": ("mai", "maithili"),
    "kok": ("kok", "gom", "konkani"),
    "doi": ("doi", "dogri"),
    "sd": ("sd", "snd", "sindhi"),
    "ks": ("ks", "kas", "kashmiri"),
    "mni": ("mni", "manipuri"),
    "brx": ("brx", "bodo"),
    "sat": ("sat", "santali"),
}
LANG = {a: code for code, al in LANG_ALIASES.items() for a in al}


def norm_lang(raw, where=""):
    """Any spelling -> one code. An unknown spelling stops the run: a language
    is one of the two balancing axes, and a guessed one is a wrong group."""
    k = str(raw).strip().lower().replace("-", "_")
    if k in LANG:
        return LANG[k]
    sys.exit(f"FATAL: language {raw!r} ({where}) is not in LANG_ALIASES in "
             f"src/train.py.\n       Add the spelling there. Nothing guesses a "
             f"language.")


# filename tokens that name a corpus or a copy, never a language
NAME_PREFIXES = {"aug", "indicsynth", "mlaad", "mms", "tts"}


def lang_from_name(stem, where):
    """First token after the corpus/copy prefixes must be a language.

      indicsynth_Bengali_00000 -> bn      mlaad_hi_<model>_00012 -> hi
      ben_tts_00000            -> bn      aug_ben_tts_00000      -> bn
    """
    toks = stem.split("_")
    i = 0
    while i < len(toks) - 1 and toks[i].lower() in NAME_PREFIXES:
        i += 1
    return norm_lang(toks[i], f"{where}: {stem}")


# --- lookup tables for names that carry no language ---------------------------
# IndicVoices chunks are <uuid>_<n>_chunk_<k> -- no language in the name at all.
# Navya's chunks.csv has it. Columns are matched loosely so her file works as
# it is: a key column (chunk/file/path/name/stem/recording_id/uuid/...), a
# language column, and optionally a speaker column.
KEY_COLS = ("chunk", "chunk_name", "chunk_id", "file", "filename", "file_name",
            "path", "name", "stem", "recording_id", "recording", "uuid", "id")
LANG_COLS = ("language", "lang", "language_code")
SPK_COLS = ("speaker", "speaker_id", "spk", "spk_id")
DUR_COLS = ("duration", "duration_s", "dur", "seconds", "secs", "length_s",
            "duration_sec")


def _pick(cols, wanted):
    low = {c.lower().strip(): c for c in cols}
    return next((low[c] for c in wanted if c in low), None)


def _key_of(v):
    return Path(str(v).strip().replace("\\", "/")).stem


def strip_chunk(stem):
    return re.sub(r"_chunk_\d+.*$", "", stem)


def load_lang_table(paths):
    """--lang-table CSV(s) -> {stem or recording id: (lang, speaker)}."""
    out = {}
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            sys.exit(f"FATAL: --lang-table {p} not found.")
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rdr = csv.DictReader(fh)
            cols = rdr.fieldnames or []
            k, l, s = _pick(cols, KEY_COLS), _pick(cols, LANG_COLS), _pick(cols, SPK_COLS)
            if not k or not l:
                sys.exit(f"FATAL: --lang-table {p} needs a key column "
                         f"({'/'.join(KEY_COLS[:6])}/...) and a language column; "
                         f"it has {cols}")
            n = 0
            for r in rdr:
                key = _key_of(r[k])
                lang = norm_lang(r[l], f"{p} row {key}")
                spk = str(r[s]).strip() if s else ""
                for kk in {key, strip_chunk(key)}:
                    prev = out.get(kk)
                    if prev and prev[0] != lang:
                        sys.exit(f"FATAL: {kk} is {prev[0]} and {lang} in "
                                 f"--lang-table. Fix the table.")
                    out[kk] = (lang, spk or (prev[1] if prev else ""))
                n += 1
        print(f"  lang-table: {n} rows from {p}"
              f"{'' if s else '  (no speaker column)'}")
    return out


def load_durations(paths):
    """--durations CSV(s) -> {stem: seconds}. A stem listed twice with different
    durations (LibriSeVoc vocoders) maps to None = unmeasured, never to a pick."""
    out = {}
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            sys.exit(f"FATAL: --durations {p} not found.")
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rdr = csv.DictReader(fh)
            cols = rdr.fieldnames or []
            k, d = _pick(cols, KEY_COLS), _pick(cols, DUR_COLS)
            if not k or not d:
                sys.exit(f"FATAL: --durations {p} needs a key column and a "
                         f"duration column ({'/'.join(DUR_COLS)}); it has {cols}")
            n = 0
            for r in rdr:
                key = _key_of(r[k])
                try:
                    v = float(r[d])
                except (TypeError, ValueError):
                    continue
                if key in out and out[key] is not None and abs(out[key] - v) > 0.05:
                    out[key] = None
                elif key not in out:
                    out[key] = v
                n += 1
        print(f"  durations: {n} rows from {p}")
    return out


# --- derive fallback -------------------------------------------------------
# Only used with --derive-groups. Every rule here is an INFERENCE from a folder
# name and a filename, which is precisely the state brief section 1 exists to
# end ("Today the metadata lives only in folder names"). It is here so the
# trainer is runnable before S1 lands, not as a substitute for S1.
DERIVE_RULES = (
    # (root substring, corpus, language, generator_family, recording_id)
    # First match wins, so the empty-substring ASVspoof catch-all stays last.
    ("indicsynth",  "indicsynth",  "from_name",  "indicsynth",  "stem"),
    ("indicvoices", "indicvoices", "from_table", "bonafide",    "speaker_or_chunk"),
    ("mms_tts",     "mms_tts",     "from_name",  "mms_tts",     "stem"),
    ("mlaad",       "mlaad",       "from_name",  "mlaad",       "stem"),
    ("librisevoc",  "librisevoc",  "en",         "from_root",   "strip_gen"),
    ("",            "asvspoof19",  "en",         "from_attack", "stem"),
)


def derive_rule_for(root: str):
    name = Path(root).resolve().name.lower()
    for sub, corpus, lang, fam, rec in DERIVE_RULES:
        if sub in name:
            return corpus, lang, fam, rec
    raise AssertionError("the empty-substring rule must always match")


def _librisevoc_family(root):
    """embeddings_v2_librisevoc_<vocoder> -> <vocoder>; ..._gt -> bonafide.

    LibriSeVoc gives every vocoder's copy of an utterance the same filename
    (<utt>_gen), so the vocoder can only come from the root. Navya's roots are
    one per vocoder for exactly this reason; files are never renamed."""
    name = Path(root).resolve().name.lower()
    tail = name.split("librisevoc", 1)[1].strip("_- ")
    if not tail:
        sys.exit(f"FATAL: {root}: a LibriSeVoc root must be named "
                 f"..._librisevoc_<vocoder> or ..._librisevoc_gt, one root per "
                 f"folder, because the filenames do not say which vocoder.")
    return "bonafide" if tail in ("gt", "real", "bonafide") else tail


def derive_group(root, stem, rule, attack_of, lang_table=None):
    corpus, lang, fam, rec = rule
    speaker = ""
    if lang == "from_name":
        lang = lang_from_name(stem, corpus)
    elif lang == "from_table":
        hit = None
        if lang_table:
            hit = lang_table.get(stem) or lang_table.get(strip_chunk(stem))
        if hit is None:
            return None                 # caller collects and stops the run
        lang, speaker = hit
    else:
        lang = norm_lang(lang, corpus)

    if fam == "from_attack":           # ASVspoof: the attack id IS the family
        fam = attack_of.get(stem, "UNKNOWN")
        if fam == "-":
            fam = "bonafide"
    elif fam == "from_root":
        fam = _librisevoc_family(root)

    if rec == "speaker_or_chunk":
        # With a speaker column, the dev carve is SPEAKER-disjoint, not just
        # recording-disjoint: IndicVoices train/holdout already leaked 682
        # speakers once, and a same-speaker dev clip scores the speaker.
        recording = f"spk:{corpus}:{speaker}" if speaker else strip_chunk(stem)
    elif rec == "strip_gen":
        # LibriSpeech names are <speaker>_<chapter>_<utt>; gt <utt> and every
        # vocoder's <utt>_gen share them. Carving on the SPEAKER keeps an
        # utterance, all its fakes and every other clip of that voice on one
        # side of the dev carve -- speaker-disjoint, like IndicVoices.
        speaker = stem.split("_")[0]
        recording = f"spk:{corpus}:{speaker}"
    else:
        recording = stem

    # the conditioned roots (g711, rawboost, rirmusan) reuse the base corpus's
    # recording ids on purpose, so a clip and its conditioned copies land on the
    # same side of the dev carve. channel is what distinguishes them.
    channel = "clean"
    for tag in ("g711", "rawboost", "rirmusan", "aug", "codec"):
        if tag in Path(root).resolve().name:
            channel = tag
            break
    return {"corpus": corpus, "language": lang, "generator_family": fam,
            "recording_id": recording, "channel": channel, "speaker": speaker}


def load_attack_ids(data_root="data/asvspoof19_la"):
    """filename -> attack id, across every LA protocol we can find.

    ASVspoof's generator family is free -- it is the attack id, already in the
    protocol. Nothing is guessed here; a file absent from every protocol comes
    back UNKNOWN and the caller decides whether that is fatal.
    """
    out = {}
    proto_dir = Path(data_root) / "ASVspoof2019_LA_cm_protocols"
    if not proto_dir.exists():
        return out
    for pf in sorted(proto_dir.glob("*.txt")):
        try:
            for line in pf.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    out[p[1]] = p[3]
        except OSError:
            continue
    return out


def attach_groups(files, y, root, manifest, derive, attack_of, strict,
                  lang_table=None):
    """Per-row group metadata for one root. Returns a list of dicts, row-aligned.

    A row whose group cannot be established stops the run. Brief section 2
    rule 4 -- nothing here estimates, defaults or fills in a group label, since
    a wrong group silently corrupts both the sampling weights and the
    worst-group metric that selects the checkpoint.
    """
    rows, unknown, no_lang, mislabel = [], [], [], []
    rule = derive_rule_for(root) if derive else None
    mrows = manifest["rows"] if manifest else None
    rname = Path(root).resolve().name

    for i, stem in enumerate(files):
        key = Path(stem).stem            # indicvoices names carry a .flac suffix
        mkey = (rname, key) if manifest and manifest["by_root"] else key
        if mrows is not None and mkey in mrows:
            m = mrows[mkey]
            # the manifest's own label must agree with the shard sidecar.
            # extract_embeddings.py:89 stamps label 0 on every --audio-dir row,
            # so a spoof-only folder comes back labelled bonafide; catching that
            # here is the difference between a balanced run and a poisoned one.
            ml = str(m["label"]).strip().lower()
            ml = 1 if ml in ("1", "spoof", "fake") else 0
            if ml != int(y[i]):
                sys.exit(
                    f"FATAL: label disagreement on {key} in {root}.\n"
                    f"       manifest says {'spoof' if ml else 'bonafide'}, shard "
                    f"sidecar says {'spoof' if y[i] else 'bonafide'}.\n"
                    f"       One of them is wrong and training on either is a\n"
                    f"       coin flip. Check whether this root was extracted\n"
                    f"       with --audio-dir, which labels every row 0.")
            rows.append({"corpus": m["corpus"], "language": m["language"],
                         "generator_family": m["generator_family"],
                         "recording_id": m["recording_id"],
                         "channel": m.get("channel", "clean"), "source": "manifest"})
        elif derive:
            g = derive_group(root, key, rule, attack_of, lang_table)
            if g is None:
                no_lang.append(key)
                rows.append(None)
                continue
            if g["generator_family"] == "UNKNOWN":
                unknown.append(key)
            # family and shard label must agree. Every --audio-dir root comes
            # out labelled 0, so a fake root that missed stamp_labels.py would
            # otherwise train as bonafide -- the same poison as above.
            elif (g["generator_family"] == "bonafide") != (int(y[i]) == 0):
                mislabel.append(key)
            g["source"] = "derived"
            rows.append(g)
        else:
            unknown.append(key)
            rows.append(None)

    if no_lang:
        sys.exit(f"FATAL: {len(no_lang)} rows in {root} have no language "
                 f"(e.g. {', '.join(no_lang[:3])}).\n"
                 f"       Their names carry none. Pass --lang-table with Navya's "
                 f"chunks.csv\n       (key column + language [+ speaker]). A "
                 f"language is never guessed.")
    if mislabel:
        n1 = int((np.asarray(y) == 1).sum())
        sys.exit(f"FATAL: {len(mislabel)} rows in {root} have a shard label "
                 f"that contradicts the folder (e.g. {', '.join(mislabel[:3])}).\n"
                 f"       Shard says {n1} spoof / {len(y) - n1} bonafide. If this "
                 f"is a FAKE root, it was\n       extracted with --audio-dir "
                 f"(labels every row 0) and never stamped:\n"
                 f"         python stamp_labels.py --emb-dir {root}\\train "
                 f"--label 1")
    if unknown:
        head = ", ".join(unknown[:5])
        if manifest is None and not derive:
            sys.exit(
                f"FATAL: no group metadata for {len(unknown)} rows in {root} "
                f"(e.g. {head}).\n"
                f"       Groups are {{language}} x {{generator_family}} x {{label}} "
                f"and none of\n"
                f"       those live in the shard sidecars. Pass --manifest with the "
                f"S1 schema\n"
                f"       ({', '.join(MANIFEST_COLUMNS)}),\n"
                f"       or --derive-groups to infer them from folder and file "
                f"names.")
        sys.exit(f"FATAL: {len(unknown)} rows in {root} have no resolvable "
                 f"generator family (e.g. {head}). They are absent from the "
                 f"manifest and from every ASVspoof protocol.")
    return rows


def duration_filter(files, y, rows, durations, min_dur):
    """-> (keep mask, [(label, n, measured, short, dropped)]).

    Short = measured duration below 4.0 s (the extractor's crop) or below
    --min-dur when given. Dropping happens only with --min-dur, and only for
    MEASURED rows; an unmeasured row is kept and shows up as n - measured."""
    thr = 4.0 if min_dur is None else min_dur
    keep = np.ones(len(files), dtype=bool)
    stats = {}
    for i, f in enumerate(files):
        d = durations.get(Path(f).stem)
        lab = int(y[i])
        st = stats.setdefault(lab, [0, 0, 0, 0])
        st[0] += 1
        if d is None:
            continue
        st[1] += 1
        if d < thr:
            st[2] += 1
            if min_dur is not None:
                keep[i] = False
                st[3] += 1
    return keep, [(lab, *v) for lab, v in sorted(stats.items())]


def report_durations(log, min_dur):
    """The padding shortcut, measured. If one class is much shorter than the
    other in some corpus, zero-padding is a label cue for that corpus."""
    thr = 4.0 if min_dur is None else min_dur
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for corpus, lab, n, meas, short, drop in log:
        a = agg[(corpus, lab)]
        for j, v in enumerate((n, meas, short, drop)):
            a[j] += v
    print(f"\n  durations (short = under {thr:g} s, i.e. zero-padded by the "
          f"extractor):")
    print(f"  {'corpus':<14}{'label':<8}{'rows':>8}{'measured':>10}"
          f"{'short':>8}{'short%':>8}{'dropped':>9}")
    for (corpus, lab), (n, meas, short, drop) in sorted(agg.items()):
        pct = f"{100 * short / meas:6.1f}%" if meas else "     -"
        print(f"  {corpus:<14}{'spoof' if lab else 'bona':<8}{n:>8}{meas:>10}"
              f"{short:>8}{pct:>8}{drop:>9}")
    unmeasured = sum(n - meas for (_c, _l), (n, meas, _s, _d) in agg.items())
    if unmeasured:
        print(f"  NOTE: {unmeasured} rows have no measured duration and were "
              f"kept as they are.")
    if min_dur is None:
        print("  (report only -- pass --min-dur 4.0 to drop the short rows)")


# ===========================================================================
# Groups, weights, and the recording-level dev carve
# ===========================================================================
def build_groups(meta_rows, y):
    """-> (group_id int array, ordered group key list, Counter of sizes).

    The group key is exactly brief section 6's: {language} x {generator_family}
    x {label}. Label is part of the key on purpose -- balancing
    (language, family) alone would leave the within-cell spoof prior untouched,
    which is the whole defect being fixed.
    """
    keys = [(m["language"], m["generator_family"], int(lab))
            for m, lab in zip(meta_rows, y)]
    order = sorted(set(keys))
    index = {k: i for i, k in enumerate(order)}
    gid = np.array([index[k] for k in keys], dtype=np.int64)
    return gid, order, Counter(keys)


def group_weights(gid, n_groups, max_weight=None):
    """w_i = N / (n_groups * n_g)  -- brief section 6 change 1, verbatim.

    Mean weight is 1.0 by construction, so the loss stays on the same scale as
    an unweighted run and --lr does not need retuning between ablations.

    max_weight exists because of brief section 8: "Weighting a 200-row cell up
    to parity overfits those 200 rows." The clamp bounds how far a starved cell
    can be amplified. It is off by default -- the brief's formula is the
    specified behaviour, and silently clamping it would make the run something
    other than what was asked for.
    """
    counts = np.bincount(gid, minlength=n_groups).astype(np.float64)
    n = float(len(gid))
    per_group = n / (n_groups * np.maximum(counts, 1.0))
    w = per_group[gid]
    if max_weight is not None:
        n_clamped = int((w > max_weight).sum())
        if n_clamped:
            print(f"  --max-weight {max_weight}: clamped {n_clamped} rows in "
                  f"{int((per_group > max_weight).sum())} starved group(s)")
        w = np.minimum(w, max_weight)
    return w.astype(np.float32), counts


def carve_dev(recording_ids, gid, n_groups, dev_frac, seed, group_langs=None):
    """Hold out dev by RECORDING, stratified so every group keeps a dev cell.

    Brief section 2 rule 3: recording-level holdout, never row-level.
    <id>_chunk_N siblings must not straddle. IndicVoices averages ~4.8 chunks
    per recording, so a row-level split there would put siblings of nearly
    every dev clip into train -- Bird & Lotfi (2023) get 99.3% that way and the
    brief is explicit that we do not reproduce the method.

    Conditioned roots (g711, rawboost, rirmusan) deliberately reuse the base
    corpus's recording ids, so a clip and its conditioned copies are the same
    recording here and land on the same side for free.

    Returns a boolean mask, True = dev.
    """
    rng = np.random.default_rng(seed)
    recs = np.asarray(recording_ids)

    # a recording belongs to one group; if its rows disagree, take the majority
    # and say so, because that means the metadata is inconsistent upstream
    by_rec = defaultdict(list)
    for i, r in enumerate(recs):
        by_rec[r].append(i)
    # A recording may legitimately span groups: LibriSeVoc's gt utterance and
    # its vocoder copies share one recording id on purpose, and so do the
    # conditioned copies. It is stratified under its RAREST group, so a starved
    # cell is never outvoted out of dev. Only a recording spanning LANGUAGES
    # means the metadata is wrong upstream.
    sizes = np.bincount(gid, minlength=n_groups)
    lang_of = group_langs if group_langs is not None else None
    rec_group, multi_lang = {}, 0
    for r, idxs in by_rec.items():
        gs = {int(gid[i]) for i in idxs}
        rec_group[r] = min(gs, key=lambda g: (sizes[g], g))
        if lang_of is not None and len({lang_of[g] for g in gs}) > 1:
            multi_lang += 1
    if multi_lang:
        print(f"  WARNING: {multi_lang} recording(s) span more than one LANGUAGE. "
              f"Chunks of one recording\n  disagree on language -- fix that "
              f"upstream (lang-table / manifest).")

    dev_recs = set()
    for g in range(n_groups):
        rs = np.array(sorted(r for r, gg in rec_group.items() if gg == g))
        if len(rs) == 0:
            continue
        k = int(round(len(rs) * dev_frac))
        # a group with recordings must contribute at least one, or its
        # worst-group EER is undefined and the selection metric silently
        # stops covering it -- which is the head_v3 failure mode
        k = max(1, min(k, len(rs) - 1)) if len(rs) > 1 else 0
        if k:
            dev_recs.update(rng.choice(rs, size=k, replace=False).tolist())

    # coverage pass: a group whose recordings were all stratified under some
    # other group can still end with no dev rows. Give it one recording.
    for g in range(n_groups):
        mine = sorted({recs[i] for i in np.flatnonzero(gid == g)})
        if len(mine) > 1 and not any(r in dev_recs for r in mine):
            dev_recs.add(mine[int(rng.integers(len(mine)))])

    mask = np.array([r in dev_recs for r in recs], dtype=bool)
    return mask


def report_groups(order, counts, gid_tr, gid_dv, n_groups):
    """Print the group table. This is the thing nobody could reconstruct for the
    nine existing heads, so it is printed AND written into the checkpoint."""
    tr = np.bincount(gid_tr, minlength=n_groups)
    dv = np.bincount(gid_dv, minlength=n_groups)
    print(f"\n  {'language':<14}{'family':<14}{'lab':>4}{'train':>8}{'dev':>7}"
          f"{'weight':>9}")
    print("  " + "-" * 56)
    n = int(tr.sum())
    starved = []
    for i, key in enumerate(order):
        w = n / (n_groups * max(int(tr[i]), 1))
        flag = ""
        if dv[i] == 0:
            flag = "  <- NO DEV CELL"
        elif tr[i] < 100:
            flag = "  <- starved"
            starved.append(key)
        print(f"  {key[0]:<14}{key[1]:<14}{key[2]:>4}{tr[i]:>8}{dv[i]:>7}"
              f"{w:>9.2f}{flag}")
    print(f"  {'TOTAL':<32}{tr.sum():>8}{dv.sum():>7}")
    if starved:
        print(f"\n  WARNING: {len(starved)} group(s) under 100 train rows are "
              f"being weighted\n  up toward parity. Brief section 8: that "
              f"overfits those few rows. Consider\n  --max-weight, a coarser "
              f"grouping, or more data for those cells.")


# ===========================================================================
# Model and losses
# ===========================================================================
def build_model(in_dim, hidden, dropout, n_families, loss, feat_dim):
    import torch
    import torch.nn as nn

    class MultitaskHead(nn.Module):
        """Shared trunk, two heads.

        The trunk is deliberately Linear -> ReLU -> Dropout and the spoof head a
        single Linear, because that makes trunk+spoof BIT-IDENTICAL in shape to
        the legacy nn.Sequential that realtime/checkpoint.py:20, src/eval.py:68
        and src/score_file.py:65 each hardcode. Those three rebuild the head
        themselves and call load_state_dict, so any other topology silently
        breaks all of serving. Under --loss bce we export a legacy state_dict
        alongside the real one and they keep working untouched.

        Under --loss ocsoftmax the spoof head is Linear(hidden, feat_dim) and
        the score is a cosine against a learned centre, which has NO equivalent
        single-logit Linear. We refuse to write a legacy key in that case rather
        than write one that loads cleanly and scores wrong -- that is the
        failure mode PROJECT_STATE section 3 documents.
        """

        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.spoof = nn.Linear(hidden, 1 if loss == "bce" else feat_dim)
            self.family = nn.Linear(hidden, n_families) if n_families > 1 else None

        def forward(self, x):
            h = self.trunk(x)
            return self.spoof(h), (self.family(h) if self.family is not None else None)

    return MultitaskHead()


def legacy_state_dict(model):
    """The v3-shaped Sequential(Linear, ReLU, Dropout, Linear) state_dict.

    Only meaningful for BCE -- see build_model's docstring. Returns None
    otherwise so the caller cannot accidentally write one.
    """
    sd = model.state_dict()
    if sd["spoof.weight"].shape[0] != 1:
        return None
    return {"0.weight": sd["trunk.0.weight"], "0.bias": sd["trunk.0.bias"],
            "3.weight": sd["spoof.weight"], "3.bias": sd["spoof.bias"]}


def make_ocsoftmax(feat_dim, m_real=0.9, m_fake=0.2, alpha=20.0):
    """OC-Softmax (Zhang et al. 2021), built for UNSEEN attacks.

    One class (bonafide) is pulled inside an angular margin around a learned
    centre; everything else is pushed outside a tighter one. Unlike BCE it does
    not model spoof as a class with its own distribution to memorise, which is
    the point for generators we have never seen.

        L = softplus( alpha * (m_y - cos(w, x)) * (-1)^y )
        bonafide (y=0): want cos >= m_real
        spoof    (y=1): want cos <= m_fake

    SIGN. cos is HIGH for bonafide. Brief section 2 rule 6 says the emitted
    score must be high for FAKE. The negation therefore happens in exactly one
    place -- score() below -- and nowhere else. Getting this backwards yields a
    ~40% EER, which CLAUDE.md lists as "almost always protocol parse or a
    flipped label"; this would be a third way to produce it.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class OCSoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.centre = nn.Parameter(torch.randn(1, feat_dim) * 0.01)
            self.m_real, self.m_fake, self.alpha = m_real, m_fake, alpha

        def cosine(self, feat):
            return F.normalize(feat, dim=1) @ F.normalize(self.centre, dim=1).t()

        def forward(self, feat, y, w):
            cos = self.cosine(feat).squeeze(1)
            margin = torch.where(y > 0.5,
                                 torch.full_like(cos, self.m_fake),
                                 torch.full_like(cos, self.m_real))
            sign = torch.where(y > 0.5, -torch.ones_like(cos), torch.ones_like(cos))
            per = F.softplus(self.alpha * (margin - cos) * sign)
            return (per * w).sum() / w.sum().clamp_min(1e-8)

        def score(self, feat):
            """Higher = FAKE. The one place the cosine is inverted."""
            return -self.cosine(feat).squeeze(1)

    return OCSoftmax()


def spoof_loss_bce(logits, y, w):
    """Weighted BCE. The per-sample group weight REPLACES pos_weight.

    pos_weight only rescales the global positive/negative ratio, so it leaves
    P(spoof | language) exactly where it was -- which is the defect. The group
    weight already carries label in its key, so class balance comes out of the
    same mechanism and a separate pos_weight would double-count it.
    """
    import torch.nn.functional as F
    per = F.binary_cross_entropy_with_logits(logits.squeeze(1), y, reduction="none")
    return (per * w).sum() / w.sum().clamp_min(1e-8)


def family_loss(logits, fam, w):
    """Weighted cross-entropy over generator family.

    Brief section 8 warns this head can learn CORPUS instead of vocoder: if each
    family appears in exactly one corpus, "which vocoder" and "which corpus" are
    the same question. The confusion matrix is written into the checkpoint so
    that can be checked rather than assumed.
    """
    import torch.nn.functional as F
    per = F.cross_entropy(logits, fam, reduction="none")
    return (per * w).sum() / w.sum().clamp_min(1e-8)


# ===========================================================================
# Worst-group EER
# ===========================================================================
def eval_cells(order, gid, y, scores, min_bona=20):
    """Per-cell EER on dev. Returns [(name, eer, n_spoof, n_bona, basis)].

    WHY THIS IS NOT THE TRAINING GROUP. The training group key is
    {language} x {generator_family} x {label}, so label is IN the key and every
    such group holds exactly one class. EER needs both, so it is undefined for
    every training group and "worst-group EER" cannot be computed over them.

    The evaluation cell is therefore coarser, and follows ASVspoof's own
    per-attack convention: each (language, family) SPOOF cell is scored against
    the bonafide of the SAME language. That is what makes the failure S7 names
    visible -- "the Indic-fake cell collapsed to zero contribution unnoticed" is
    a statement about one (language, family) cell, not about a single-class
    group. When a language has fewer than min_bona bonafide rows in dev we fall
    back to the whole bonafide pool and label the cell 'all-bona', because a
    cell scored against 3 bonafide clips is noise, not a metric.
    """
    lang_of = {i: k[0] for i, k in enumerate(order)}
    fam_of = {i: k[1] for i, k in enumerate(order)}
    lab_of = {i: k[2] for i, k in enumerate(order)}

    bona_mask = y == 0
    out = []
    for i, key in enumerate(order):
        if lab_of[i] != 1:
            continue                       # bonafide cells are the reference side
        sel = gid == i
        n_spoof = int(sel.sum())
        if n_spoof == 0:
            continue
        same_lang = bona_mask & np.array([lang_of.get(g) == lang_of[i]
                                          for g in gid])
        if int(same_lang.sum()) >= min_bona:
            ref, basis = same_lang, "same-lang"
        else:
            ref, basis = bona_mask, "all-bona"
        n_bona = int(ref.sum())
        if n_bona == 0:
            out.append((f"{lang_of[i]}/{fam_of[i]}", float("nan"), n_spoof, 0,
                        "no-bona"))
            continue
        m = sel | ref
        out.append((f"{lang_of[i]}/{fam_of[i]}",
                    eer(y[m], scores[m]), n_spoof, n_bona, basis))
    return out


def summarise(cells):
    """(worst, mean, n_valid). NaN cells are excluded and counted by the caller."""
    vals = [c[1] for c in cells if not np.isnan(c[1])]
    if not vals:
        return float("nan"), float("nan"), 0
    return max(vals), float(np.mean(vals)), len(vals)


# ===========================================================================
# CLI
# ===========================================================================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Train the SONIX v4 head: group-balanced, multitask, "
                    "selected on worst-group EER.")

    # --- roots. Teammates' scripts pass these; the names do not change. ---
    ap.add_argument("--emb-root", default="outputs/embeddings")
    ap.add_argument("--extra-emb-root", default=None, action="append",
                    metavar="ROOT",
                    help="extra embeddings root concatenated into training. "
                         "REPEATABLE. Every root is checked against the held-out "
                         "IndicVoices set before anything is read.")
    ap.add_argument("--layer", default=None,
                    help="slice ONE block out of a v2 root, by its meta.json "
                         "label (an int, or 'ln'). Omit to use the full vector.")

    # --- group metadata ---
    ap.add_argument("--manifest", default=None,
                    help="S1 manifest CSV with columns: "
                         + ", ".join(MANIFEST_COLUMNS)
                         + ". The preferred and only non-inferred source.")
    ap.add_argument("--derive-groups", action="store_true",
                    help="infer language/family/recording_id from folder and "
                         "file names when no manifest exists. A STOPGAP -- see "
                         "the warning it prints.")
    ap.add_argument("--lang-table", default=None, action="append", metavar="CSV",
                    help="REPEATABLE. CSV giving language (and optionally "
                         "speaker) for names that carry none -- IndicVoices. "
                         "Navya's chunks.csv works as is: any key column "
                         "(chunk/file/path/name/stem/recording_id/uuid) plus "
                         "language [+ speaker]. With a speaker column the "
                         "IndicVoices dev carve is speaker-disjoint.")
    ap.add_argument("--durations", default=None, action="append", metavar="CSV",
                    help="REPEATABLE. CSV of clip durations (key column + "
                         "duration/seconds). Always reported per corpus x label; "
                         "used to drop rows with --min-dur.")
    ap.add_argument("--min-dur", type=float, default=None,
                    help="drop rows whose MEASURED duration is below this many "
                         "seconds (4.0 = the extractor's crop). Clips under 4 s "
                         "are zero-padded, and IndicVoices is 36%% short against "
                         "~0%% for IndicSynth, so padding alone would read as "
                         "'real'. Unmeasured rows are kept and counted.")
    ap.add_argument("--data-root", default="data/asvspoof19_la",
                    help="for the CM protocols, which give ASVspoof its "
                         "generator family (the attack id) under --derive-groups")

    # --- the three v4 changes ---
    ap.add_argument("--loss", default="bce", choices=["bce", "ocsoftmax"])
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="weight on the generator-family head. 0 disables it.")
    ap.add_argument("--feat-dim", type=int, default=128,
                    help="OC-Softmax embedding width (ignored for BCE)")
    ap.add_argument("--max-weight", type=float, default=None,
                    help="clamp the per-group upweight. Off by default: the "
                         "brief specifies N/(n_groups*n_g) and clamping it "
                         "silently would make this a different run.")

    # --- the dev carve ---
    ap.add_argument("--dev-frac", type=float, default=0.15,
                    help="fraction of RECORDINGS held out as dev, per group")
    ap.add_argument("--dev-seed", type=int, default=0)
    ap.add_argument("--min-bona", type=int, default=20,
                    help="below this many same-language bonafide rows in dev, a "
                         "cell is scored against the whole bonafide pool instead")

    # --- the usual ---
    ap.add_argument("--out", default="outputs/models/head_v4.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--device", default=None)
    return ap


def main(argv=None) -> int:
    import torch

    args = build_argparser().parse_args(argv)
    if args.manifest is None and not args.derive_groups:
        sys.exit(
            "FATAL: no group source. v4 balances over {language} x "
            "{generator_family} x {label},\n"
            "       and none of those live in the shard sidecars.\n"
            "       Pass --manifest <S1 csv>, or --derive-groups to infer them "
            "from folder\n"
            "       and file names. There is no default, because guessing a "
            "group silently\n"
            "       corrupts both the sampling weights and the metric that "
            "picks the checkpoint.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   loss: {args.loss}   aux-weight: {args.aux_weight}")

    manifest = load_manifest(args.manifest) if args.manifest else None
    attack_of = load_attack_ids(args.data_root) if args.derive_groups else {}
    lang_table = load_lang_table(args.lang_table)
    durations = load_durations(args.durations)
    if args.derive_groups:
        print("  --derive-groups: inferring metadata from folder and file names.\n"
              "  This is the state brief section 1 exists to end. Build the S1 "
              "manifest and\n"
              "  pass --manifest as soon as it exists; the checkpoint records "
              "which source\n"
              "  was used, so a run can never be mistaken for the other.")

    # ---- load every root -------------------------------------------------
    roots = [args.emb_root] + list(args.extra_emb_root or [])
    print(f"\nloading {len(roots)} root(s):")
    Xs, ys, gm, prov, dur_log = [], [], [], [], []
    for r in roots:
        X, y, files, meta = load_root(r, "train", args.layer)
        rows = attach_groups(files, y, r, manifest, args.derive_groups,
                             attack_of, strict=True, lang_table=lang_table)
        n_in = len(X)
        dur_stats = None
        if durations:
            keep, dur_stats = duration_filter(files, y, rows, durations,
                                              args.min_dur)
            dur_log += [(rows[0]["corpus"], *t) for t in dur_stats]
            if not keep.all():
                X, y = X[keep], y[keep]
                rows = [g for g, k in zip(rows, keep) if k]
        recs = [d["recording_id"] for d in rows]
        Xs.append(X)
        ys.append(y)
        gm += rows
        prov.append({"root": r, "n": int(len(X)), "n_before_min_dur": int(n_in),
                     "meta": meta, "n_recordings": len(set(recs)),
                     "durations": dur_stats})
    if dur_log:
        report_durations(dur_log, args.min_dur)

    dims = {x.shape[1] for x in Xs}
    if len(dims) > 1:
        sys.exit(f"FATAL: roots disagree on dimensionality: {sorted(dims)}.\n"
                 f"       They came from different front-end configurations and "
                 f"cannot be\n"
                 f"       concatenated. Use --layer to slice them to a common "
                 f"block.")

    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    recordings = [d["recording_id"] for d in gm]
    del Xs, ys

    # ---- groups, then the recording-level carve --------------------------
    gid, order, _ = build_groups(gm, y)
    n_groups = len(order)
    print(f"\n{len(X)} rows, {n_groups} training groups, "
          f"{len(set(recordings))} recordings")

    dev_mask = carve_dev(recordings, gid, n_groups, args.dev_frac, args.dev_seed,
                         group_langs=[k[0] for k in order])
    tr_mask = ~dev_mask
    if dev_mask.sum() == 0:
        sys.exit("FATAL: the dev carve is empty. Raise --dev-frac.")

    # the leak check, run rather than assumed
    tr_recs = set(np.asarray(recordings)[tr_mask])
    dv_recs = set(np.asarray(recordings)[dev_mask])
    straddle = tr_recs & dv_recs
    if straddle:
        sys.exit(f"FATAL: {len(straddle)} recording(s) appear in BOTH train and "
                 f"dev, e.g. {sorted(straddle)[:3]}. Brief section 2 rule 3.")
    print(f"  carve: {int(tr_mask.sum())} train / {int(dev_mask.sum())} dev rows; "
          f"{len(tr_recs)}/{len(dv_recs)} recordings, 0 straddling")

    Xtr, ytr, gtr = X[tr_mask], y[tr_mask], gid[tr_mask]
    Xdv, ydv, gdv = X[dev_mask], y[dev_mask], gid[dev_mask]
    del X

    report_groups(order, None, gtr, gdv, n_groups)

    # ---- weights, from TRAIN counts only ---------------------------------
    w_tr, _ = group_weights(gtr, n_groups, args.max_weight)
    print(f"\n  group weights: min {w_tr.min():.3f}  max {w_tr.max():.3f}  "
          f"mean {w_tr.mean():.3f}  (mean is 1.0 by construction)")

    # ---- family labels for the auxiliary head ----------------------------
    fams = sorted({k[1] for k in order})
    fam_index = {f: i for i, f in enumerate(fams)}
    fam_tr = np.array([fam_index[order[g][1]] for g in gtr], np.int64)
    n_families = len(fams)
    use_aux = args.aux_weight > 0 and n_families > 1
    if args.aux_weight > 0 and not use_aux:
        print(f"  NOTE: --aux-weight {args.aux_weight} but only {n_families} "
              f"generator family present; the auxiliary head is disabled.")
    print(f"  generator families ({n_families}): {', '.join(fams)}")

    # ---- standardiser, train stats only ----------------------------------
    if args.no_standardize:
        mu = np.zeros(Xtr.shape[1], np.float32)
        sd = np.ones(Xtr.shape[1], np.float32)
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xdv = (Xdv - mu) / sd

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    wtr_t = torch.from_numpy(w_tr).float()
    ftr_t = torch.from_numpy(fam_tr)
    Xdv_t = torch.from_numpy(Xdv).float().to(device)

    model = build_model(Xtr.shape[1], args.hidden, args.dropout,
                        n_families, args.loss, args.feat_dim).to(device)
    oc = (make_ocsoftmax(args.feat_dim).to(device)
          if args.loss == "ocsoftmax" else None)
    params = list(model.parameters()) + (list(oc.parameters()) if oc else [])
    print(f"  head parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.Adam(params, lr=args.lr)

    # ---- train -----------------------------------------------------------
    n = len(Xtr_t)
    best = {"worst": float("inf"), "epoch": -1, "state": None, "oc": None,
            "cells": None, "mean": float("nan")}
    since = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        run_s = run_a = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            wb = wtr_t[idx].to(device)
            opt.zero_grad()
            sp, fam_logits = model(xb)
            ls = (oc(sp, yb, wb) if oc is not None
                  else spoof_loss_bce(sp, yb, wb))
            loss = ls
            la = 0.0
            if use_aux:
                la = family_loss(fam_logits, ftr_t[idx].to(device), wb)
                loss = loss + args.aux_weight * la
            loss.backward()
            opt.step()
            run_s += float(ls) * len(idx)
            run_a += float(la) * len(idx)

        model.eval()
        with torch.no_grad():
            sp, _ = model(Xdv_t)
            dev_scores = (oc.score(sp) if oc is not None
                          else sp.squeeze(1)).cpu().numpy()
        cells = eval_cells(order, gdv, ydv, dev_scores, args.min_bona)
        worst, mean, n_valid = summarise(cells)
        print(f"epoch {epoch:2d}/{args.epochs}  spoof={run_s / n:.4f}  "
              f"fam={run_a / n:.4f}  worst-group EER={worst * 100:.2f}%  "
              f"mean={mean * 100:.2f}%  ({n_valid} cells)")

        if worst < best["worst"] - 1e-5:
            best.update(worst=worst, mean=mean, epoch=epoch, cells=cells,
                        state={k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()},
                        oc=({k: v.detach().cpu().clone()
                             for k, v in oc.state_dict().items()}
                            if oc else None))
            since = 0
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop: no worst-group improvement for "
                      f"{args.patience} epochs")
                break

    if best["state"] is None:
        sys.exit("FATAL: no epoch produced a finite worst-group EER. Every dev "
                 "cell was single-class or empty -- check the carve and the "
                 "group table above.")
    model.load_state_dict(best["state"])
    if oc is not None:
        oc.load_state_dict(best["oc"])

    print(f"\nBEST worst-group EER = {best['worst'] * 100:.2f}%  "
          f"(mean {best['mean'] * 100:.2f}%, epoch {best['epoch']})")
    print(f"\n  {'cell':<28}{'EER':>8}{'spoof':>8}{'bona':>7}  basis")
    for name, e, ns, nb, basis in sorted(best["cells"], key=lambda c: -c[1]):
        print(f"  {name:<28}{e * 100:>7.2f}%{ns:>8}{nb:>7}  {basis}")
    print("\n  Worst-group is the selection metric. The mean is printed beside "
          "it because\n  brief section 8 warns worst-group selection can drag "
          "the mean down; if it\n  has, back off to a weighted combination "
          "rather than shipping this.")

    # ---- is this metric actually measuring detection? --------------------
    # A spoof cell scored against POOLED bonafide instead of same-language
    # bonafide is, in this representation, largely a corpus test:
    # docs/PROBE_RESULTS.md probe B separates IndicVoices from ASVspoof at
    # 100.00% among same-label rows. So "Indic spoof vs English bonafide" can
    # sit near 0% EER while the head has learned no synthesis signal at all.
    # Printing that number without this warning is how a shortcut ships.
    fell_back = [c for c in best["cells"] if c[4] != "same-lang"]
    if fell_back:
        print(f"\n  WARNING: {len(fell_back)} of {len(best['cells'])} cells were "
              f"scored against the POOLED\n  bonafide, not same-language "
              f"bonafide ({args.min_bona}+ rows required). For those cells the\n"
              f"  EER is a corpus test as much as a detection test -- probe B "
              f"in\n  docs/PROBE_RESULTS.md separates the corpora at 100.00% "
              f"among same-label rows,\n  so a near-zero EER here is expected "
              f"whether or not the head learned anything\n  about synthesis. "
              f"Treat these as UNMEASURED:")
        for name, e, ns, nb, basis in sorted(fell_back)[:8]:
            print(f"    {name:<26} {e * 100:>6.2f}%  ({basis})")
        if len(fell_back) > 8:
            print(f"    ... and {len(fell_back) - 8} more")
        print("  Fix is data, not code: same-language bonafide for these "
              "languages (S2),\n  with language in the manifest (S1).")

    undet = sorted({k[0] for k in order if k[0] in ("und", "unknown", "")})
    if undet:
        n_und = int(sum((gtr == i).sum() for i, k in enumerate(order)
                        if k[0] in undet))
        print(f"\n  WARNING: {n_und} train rows have an undetermined language. "
              f"Language is one of\n  the two balancing axes, so those rows form "
              f"a single degenerate cell and\n  cannot pair with any "
              f"same-language spoof cell. Under --derive-groups this\n  is "
              f"expected for IndicVoices, whose filenames carry no language at "
              f"all\n  (1407374883615481_chunk_2.flac). The S1 manifest is what "
              f"fixes it.")

    # ---- checkpoint ------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "head_type": "multitask_" + args.loss,
        "mu": mu, "sd": sd,
        "config": {"in_dim": int(Xtr.shape[1]), "hidden": args.hidden,
                   "dropout": args.dropout, "loss": args.loss,
                   "feat_dim": args.feat_dim, "n_families": n_families,
                   "standardized": not args.no_standardize},
        "families": fams,
        "groups": [{"language": k[0], "generator_family": k[1], "label": k[2],
                    "n_train": int((gtr == i).sum()),
                    "n_dev": int((gdv == i).sum())}
                   for i, k in enumerate(order)],
        "dev_worst_group_eer": best["worst"],
        "dev_mean_group_eer": best["mean"],
        "dev_cells": [{"cell": c[0], "eer": c[1], "n_spoof": c[2],
                       "n_bona": c[3], "basis": c[4]} for c in best["cells"]],
        # provenance: nine heads exist and nobody can reconstruct what any of
        # them saw (brief section 6). That ends here.
        "provenance": {"roots": prov, "layer": args.layer,
                       "group_source": "manifest" if manifest else "derived",
                       "manifest": args.manifest,
                       "lang_table": args.lang_table,
                       "durations": args.durations, "min_dur": args.min_dur,
                       "dev_frac": args.dev_frac, "dev_seed": args.dev_seed,
                       "argv": sys.argv[1:]},
        "front_end": "facebook/wav2vec2-xls-r-300m",
    }
    if oc is not None:
        ckpt["ocsoftmax"] = oc.state_dict()

    lsd = legacy_state_dict(model)
    if lsd is not None:
        ckpt["legacy_state_dict"] = lsd
        print("\n  legacy_state_dict written: src/eval.py, src/score_file.py "
              "and realtime/\n  load this key unchanged.")
    else:
        print("\n  NO legacy_state_dict. --loss ocsoftmax scores by cosine to a "
              "learned centre,\n  which has no equivalent Linear(hidden,1), so "
              "eval.py / score_file.py /\n  realtime/ need updating before this "
              "checkpoint can serve. Writing a key\n  they could load and score "
              "wrong with is the worse option.")

    torch.save(ckpt, out)
    print(f"\nsaved -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
