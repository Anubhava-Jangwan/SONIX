#!/usr/bin/env python3
"""
probe_layers.py  --  SONIX / SIH26104        S4 step 2

Re-runs probes A, B and C ONCE PER LAYER on the extract_embeddings_v2 sweep, so
the v4 front-end layer is chosen by measurement instead of by the convention
that last_hidden_state is the useful one.

WHY
    probe_shortcut.py / probe_transfer.py (2026-09-19, locked in
    docs/PROBE_RESULTS.md) measured ONE representation -- frozen XLS-R,
    last_hidden_state, mean(dim=1):

        A  language, 12-way, within IndicSynth only ......  99.33%  (chance 8.33%)
        B  corpus, bonafide only, recording-split ........ 100.00%  (chance 82.54%)
        C  spoof probe trained EN, tested Indic .......... AUC 0.5196 (chance 0.50)

    That is one point in a 26-way design space. XLS-R has 25 hidden states plus
    the post-final-layernorm top, and the sweep stores mean AND std for every
    one of them. This script reads all 26 and asks, per block: how decodable is
    language, how decodable is corpus, and how much spoof signal survives to an
    unseen generator.

    The pick is the block with the LOWEST A and B and the HIGHEST C-unseen.

REUSED, NOT REWRITTEN
    load_root()  <- probe_shortcut.py:35   shard concat + files.txt alignment
    fit_probe()  <- probe_shortcut.py:54   group split, scaler, logreg, chance
    halves()     <- probe_transfer.py:84   50/50 class-aligned split
    the AUC body <- probe_transfer.py:94   evaluate()

    The probe statistics are unchanged, so a per-block number here is directly
    comparable to the locked numbers computed the same way.

WHAT IS NOT THE SAME AS THE LOCKED PROBES -- read before quoting a number
    The locked probes used IndicVoices (Indic bonafide) and IndicSynth (Indic
    spoof, 12 languages). Neither has audio on this machine:
    docs/V4_BUILD_BRIEF.md section 4 lists IndicVoices as "re-download from
    AI4Bharat" and the IndicSynth wavs as living on one teammate's machine, and
    embeddings cannot be re-pooled. The only Indic audio on disk is
    data/indic_spoof -- 800 MMS-TTS clips, 5 languages, all spoof. So:

      A  is 5-way (ben/hin/mar/tam/tel) within data/indic_spoof, chance 20%,
         NOT 12-way within IndicSynth, chance 8.33%.
         CAVEAT: MMS-TTS is one voice per language, so language and speaker are
         confounded here -- 5-way language ID is also 5-way voice ID. The locked
         12-way probe had the same property (one generator family).

      B  holds the label constant on the SPOOF side (ASVspoof spoof vs
         indic_spoof spoof), because there is no Indic bonafide to hold it
         constant on the bonafide side as the locked B did.

      C  is leave-attack-family-out INSIDE ASVspoof: fit bonafide vs A01-A04,
         test bonafide vs A05+A06. That is unseen-GENERATOR transfer, in-corpus.
         It is a weaker claim than the locked cross-lingual C and must never be
         quoted as a delta against 0.5196. Pass --indic-bonafide <root> once
         IndicVoices audio is re-acquired (S2) and the real cross-lingual C runs
         instead, with no other change to this script.

    None of the three is a drop-in replacement for a locked number. They are an
    internally consistent per-block comparison whose baseline row is the "ln"
    block -- bit-identical to what v1 and realtime/frontend.py serve, verified
    on 50 train clips at max abs diff 0.000000. Every delta in the output table
    is against that row, on the same clips, not against the locked figures.

INPUTS
    outputs/emb_v2_sweep/train              ASVspoof19 LA train, --layers all,ln
    outputs/emb_v2_sweep_indicfake/train    data/indic_spoof,    --layers all,ln
    data/asvspoof19_la/...cm.train.trn.txt  the attack ids probe C needs

RUN
    .venv\\Scripts\\python.exe probe_layers.py
    .venv\\Scripts\\python.exe probe_layers.py --blocks 9,14,ln    # subset, fast
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 0

PROTOCOL = os.path.join(ROOT, "data", "asvspoof19_la",
                        "ASVspoof2019_LA_cm_protocols",
                        "ASVspoof2019.LA.cm.train.trn.txt")

SEEN_ATTACKS = ("A01", "A02", "A03", "A04")     # probe C: fitted on these
UNSEEN_ATTACKS = ("A05", "A06")                 # probe C: tested on these


# ===========================================================================
# Loading
# ===========================================================================
class Sweep:
    """One extract_embeddings_v2 root: the float16 matrix, filenames, meta.

    This is probe_shortcut.load_root() with two changes, both forced:

    1. X STAYS float16. The reference loader casts the whole matrix to float32
       up front. Here the matrix is 26 x 2048 wide -- 6000 clips is 639 MB in
       float16 and 1.28 GB in float32, and every probe only ever touches one
       2048-dim block. So the cast happens at slice time, on ~49 MB, not here.

    2. LABELS CAN BE FORCED. extract_embeddings.py:89 (_build_manifest_dir)
       stamps label 0 on every file when --audio-dir is used, so the
       data/indic_spoof shards come back labelled BONAFIDE. They are all spoof.
       Any root loaded with force_label= overrides the sidecar, which is
       otherwise a silent label flip straight into probe B.
    """

    def __init__(self, rel, split="train", force_label=None):
        d = os.path.join(ROOT, rel, split)
        Xs, ys, fs = [], [], []
        for npy in sorted(glob.glob(os.path.join(d, "*.npy"))):
            if npy.endswith(".labels.npy"):
                continue
            stem = npy[:-4]
            a = np.load(npy)
            if a.shape[0] == 0:          # empty shard: a failed batch, skipped
                continue
            Xs.append(a)
            ys.append(np.load(stem + ".labels.npy"))
            with open(stem + ".files.txt") as fh:
                fs += [ln.strip() for ln in fh if ln.strip()]
        if not Xs:
            sys.exit(f"FATAL: no shards in {d}\n"
                     f"       run src/extract_embeddings_v2.py --layers all,ln first")

        self.X = np.concatenate(Xs)                     # float16, (n, 26*P)
        self.y = np.concatenate(ys).astype(int)
        self.files = fs
        self.rel = rel
        assert len(fs) == len(self.X), f"{rel}: {len(fs)} names vs {len(self.X)} rows"

        meta_p = os.path.join(d, "meta.json")
        if not os.path.exists(meta_p):
            sys.exit(f"FATAL: {meta_p} missing -- that sidecar carries the block "
                     f"labels, and without it a slice cannot be named.")
        self.meta = json.loads(open(meta_p).read())
        self.P = self.meta["dim_per_layer"]
        self.labels = self.meta["layers"]               # ints, plus "ln"

        want = len(self.labels) * self.P
        if self.X.shape[1] != want:
            sys.exit(f"FATAL: {rel} is {self.X.shape[1]} dims but meta.json "
                     f"declares {len(self.labels)} blocks x {self.P} = {want}")

        if force_label is not None:
            self.y = np.full(len(self.X), int(force_label))

        tag = (f"[label forced to {force_label}]" if force_label is not None
               else f"[spoof={int((self.y == 1).sum())} "
                    f"bona={int((self.y == 0).sum())}]")
        print(f"  {rel:38s} n={len(self.X):6d}  dim={self.X.shape[1]:6d}  "
              f"blocks={len(self.labels)}  {tag}")

    def block(self, label):
        """Slice one block out as float32. `label` is an int or "ln" -- the
        POSITION in meta['layers'], not the raw layer number, which is why this
        goes through .index() rather than arithmetic on the label."""
        i = self.labels.index(label)
        return self.X[:, i * self.P:(i + 1) * self.P].astype(np.float32)


def load_attack_ids():
    """filename -> attack id ('-' for bonafide), from the CM protocol.

    Probe C needs to split spoof rows by generator family, and the shard
    sidecars carry only a binary label. The protocol is the only place the
    attack id exists.
    """
    if not os.path.exists(PROTOCOL):
        sys.exit(f"FATAL: protocol not found: {PROTOCOL}\n"
                 f"       probe C splits by attack id and cannot run without it.")
    m = {}
    with open(PROTOCOL) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 5:
                m[p[1]] = p[3]
    return m


# ===========================================================================
# Probe machinery -- lifted from the two reference scripts, statistics intact
# ===========================================================================
def fit_probe(X, y, groups=None, seed=SEED):
    """probe_shortcut.fit_probe(), returning instead of printing.

    Unchanged: the 0.25 group-or-stratified split, StandardScaler fitted on
    train only, LogisticRegression(max_iter=1000), accuracy against the
    majority-class baseline. 78 fits cannot each print three lines, so the
    caller formats.
    """
    if groups is not None:
        uniq = np.unique(groups)
        tr_g, te_g = train_test_split(uniq, test_size=0.25, random_state=seed)
        tr, te = np.isin(groups, tr_g), np.isin(groups, te_g)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    # the reference scripts pass n_jobs=-1; it became a no-op in scikit-learn
    # 1.8 (this machine runs 1.9) and now emits a FutureWarning per fit, which
    # is 78 warnings in one run. Dropping it changes no result.
    clf = LogisticRegression(max_iter=1000)
    clf.fit(sc.transform(Xtr), ytr)
    acc = clf.score(sc.transform(Xte), yte)
    chance = np.bincount(yte).max() / len(yte)
    return acc, chance, len(ytr), len(yte)


def halves(A, B):
    """probe_transfer.halves() verbatim -- split two class arrays 50/50 into
    train and test, keeping labels aligned."""
    ia, ib = len(A) // 2, len(B) // 2
    Xtr = np.vstack([A[:ia], B[:ib]])
    ytr = np.r_[np.zeros(ia, int), np.ones(ib, int)]
    Xte = np.vstack([A[ia:], B[ib:]])
    yte = np.r_[np.zeros(len(A) - ia, int), np.ones(len(B) - ib, int)]
    return Xtr, ytr, Xte, yte


def subsample(X, n, rng):
    """probe_transfer.subsample(), with the rng passed in so each probe is
    reproducible independently of call order."""
    if len(X) <= n:
        return X
    return X[rng.choice(len(X), size=n, replace=False)]


# ---------------------------------------------------------------- PROBE A
def probe_a(indic, label):
    """LANGUAGE, 5-way, within data/indic_spoof only.

    Every row is spoof, Indic, and MMS-TTS, so nothing varies except language
    -- the same control structure as the locked 12-way IndicSynth probe, and
    the same caveat: one voice per language, so this is also voice ID.

    LOWER IS BETTER. A block that cannot read language off the vector is a
    block where "it is Hindi" is not available as a shortcut to the head.
    """
    X = indic.block(label)
    langs = np.array([f.split("_")[0] for f in indic.files])
    names = sorted(set(langs))
    y = np.array([names.index(v) for v in langs])
    acc, chance, ntr, nte = fit_probe(X, y)
    return {"acc": acc, "chance": chance, "n_train": ntr, "n_test": nte,
            "classes": names}


# ---------------------------------------------------------------- PROBE B
def probe_b(en, indic, label, rng):
    """CORPUS, with the label held constant on the SPOOF side.

    ASVspoof spoof vs indic_spoof spoof. Both sides label=1, so the label
    cannot explain the separation. Balanced to the smaller side.

    The locked B held the label constant on the BONAFIDE side; there is no
    Indic bonafide on this machine to do that with. Same statistic, different
    rows -- do not quote this against 100.00%.

    LOWER IS BETTER.
    """
    Xe_all = en.block(label)
    Xe = Xe_all[en.y == 1]
    Xi = indic.block(label)
    n = min(len(Xe), len(Xi))
    Xe, Xi = subsample(Xe, n, rng), subsample(Xi, n, rng)
    X = np.vstack([Xe, Xi])
    y = np.r_[np.zeros(len(Xe), int), np.ones(len(Xi), int)]
    acc, chance, ntr, nte = fit_probe(X, y)
    return {"acc": acc, "chance": chance, "n_train": ntr, "n_test": nte,
            "n_per_side": n}


# ---------------------------------------------------------------- PROBE C
def probe_c(en, label, attack_of):
    """TRANSFER to an unseen generator family, inside ASVspoof.

    Fit bonafide vs A01-A04. Test twice, against the SAME held-out bonafide
    half, changing only which spoofs appear:
        AUC(seen)   bonafide[half:] vs A01-A04[half:]
        AUC(unseen) bonafide[half:] vs A05 + A06

    So the gap between the two columns is attributable to the generator family
    and to nothing else -- same corpus, same channel, same language, same
    bonafide rows. That is the whole point: it isolates synthesis signal from
    the stationary corpus fingerprint the locked probes showed the vector is
    full of.

    HIGHER IS BETTER on the unseen column. This is the criterion the front-end
    layer is picked on.

    NOT the locked cross-lingual probe C. AUC 0.5196 was English-fit tested on
    Indic. This is English-fit tested on English, unseen attacks. Weaker claim,
    and the only one the audio on disk supports today.
    """
    X = en.block(label)
    atk = np.array([attack_of.get(f, "?") for f in en.files])
    bona = X[en.y == 0]
    seen = X[np.isin(atk, SEEN_ATTACKS)]
    unseen = X[np.isin(atk, UNSEEN_ATTACKS)]

    # balance the fitted cells. probe_transfer.py feeds halves() cells that are
    # already equal (2580 each); ASVspoof train is ~10% bonafide, so handing it
    # 650 vs 4500 would be a different fit from the reference one. AUC itself is
    # prevalence-invariant, but the decision boundary is not, and the whole
    # point is that this column stays comparable across blocks and to the
    # reference method. Same rng per call, so every block sees the same rows.
    rng_c = np.random.default_rng(SEED)
    seen = subsample(seen, len(bona), rng_c)

    Xtr, ytr, Xte, yte = halves(bona, seen)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)

    bona_te = bona[len(bona) // 2:]
    Xun = np.vstack([bona_te, unseen])
    yun = np.r_[np.zeros(len(bona_te), int), np.ones(len(unseen), int)]

    auc_seen = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])
    auc_unseen = roc_auc_score(yun, clf.predict_proba(sc.transform(Xun))[:, 1])
    return {"auc_seen": auc_seen, "auc_unseen": auc_unseen,
            "gap": auc_seen - auc_unseen,
            "n_bona": len(bona), "n_seen": len(seen), "n_unseen": len(unseen)}


def probe_c_crosslingual(en, indic_real, indic_fake, label, rng, n_cell):
    """The LOCKED probe C, per block -- runs only when --indic-bonafide is given.

    probe_transfer.evaluate() applied to one slice: fit spoof/bonafide on
    English, test in-domain and on Indic. Identical statistic to the run that
    produced AUC 0.5196, so THIS column is comparable to the locked number and
    the leave-attack-out columns are not.

    Blocked until IndicVoices audio is re-acquired (S2). Nothing calls it until
    then, and it is written now so that acquiring the audio needs no code change.
    """
    Xe = en.block(label)
    en_real = subsample(Xe[en.y == 0], n_cell, rng)
    en_fake = subsample(Xe[en.y == 1], n_cell, rng)
    in_real = subsample(indic_real.block(label), n_cell, rng)
    in_fake = subsample(indic_fake.block(label), n_cell, rng)

    Xtr, ytr, Xen, yen = halves(en_real, en_fake)
    Xin = np.vstack([in_real, in_fake])
    yin = np.r_[np.zeros(len(in_real), int), np.ones(len(in_fake), int)]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    return {"auc_en": roc_auc_score(yen, clf.predict_proba(sc.transform(Xen))[:, 1]),
            "auc_in": roc_auc_score(yin, clf.predict_proba(sc.transform(Xin))[:, 1])}


# ===========================================================================
# Driver
# ===========================================================================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Per-layer probes A/B/C on an extract_embeddings_v2 sweep.")
    ap.add_argument("--en-root", default="outputs/emb_v2_sweep",
                    help="ASVspoof sweep root (has train/ with meta.json)")
    ap.add_argument("--indic-fake-root", default="outputs/emb_v2_sweep_indicfake",
                    help="data/indic_spoof sweep root; labels are FORCED to spoof")
    ap.add_argument("--indic-bonafide", default=None,
                    help="Indic bonafide sweep root. When given, the LOCKED "
                         "cross-lingual probe C runs as an extra column. Blocked "
                         "on S2 until IndicVoices audio is re-acquired.")
    ap.add_argument("--blocks", default="all",
                    help="'all', or a comma list of block labels, e.g. '9,14,ln'")
    ap.add_argument("--n-cell", type=int, default=2580,
                    help="cell size for the cross-lingual probe C "
                         "(2580 matches the locked run)")
    ap.add_argument("--out", default="outputs/probe_layers.md",
                    help="markdown table; a .csv is written alongside it")
    return ap


def main():
    args = build_argparser().parse_args()
    rng = np.random.default_rng(SEED)

    print("loading sweeps ...")
    en = Sweep(args.en_root)
    indic = Sweep(args.indic_fake_root, force_label=1)
    indic_real = Sweep(args.indic_bonafide, force_label=0) if args.indic_bonafide else None

    if en.labels != indic.labels or en.P != indic.P:
        sys.exit(f"FATAL: the two sweeps disagree on block layout.\n"
                 f"  {en.rel}: {en.labels}\n  {indic.rel}: {indic.labels}\n"
                 f"       both must be extracted with the same --layers/--pool.")

    attack_of = load_attack_ids()
    atk = np.array([attack_of.get(f, "?") for f in en.files])
    miss = int((atk == "?").sum())
    if miss:
        sys.exit(f"FATAL: {miss} of {len(en.files)} ASVspoof filenames are absent "
                 f"from the protocol; probe C cannot split them by attack family.")
    n_seen = int(np.isin(atk, SEEN_ATTACKS).sum())
    n_unseen = int(np.isin(atk, UNSEEN_ATTACKS).sum())
    n_bona = int((en.y == 0).sum())
    print(f"  attack split: bonafide={n_bona}  "
          f"seen{list(SEEN_ATTACKS)}={n_seen}  unseen{list(UNSEEN_ATTACKS)}={n_unseen}")
    for nm, v in (("bonafide", n_bona), ("seen-attack", n_seen),
                  ("unseen-attack", n_unseen)):
        if v < 100:
            print(f"  WARNING: only {v} {nm} rows. Extract a larger --limit "
                  f"before trusting probe C.")

    if args.blocks.strip().lower() == "all":
        blocks = list(en.labels)
    else:
        blocks = [t.strip() for t in args.blocks.split(",") if t.strip()]
        blocks = [t if t == "ln" else int(t) for t in blocks]
        bad = [b for b in blocks if b not in en.labels]
        if bad:
            sys.exit(f"FATAL: no such block(s) {bad}; sweep has {en.labels}")

    xl = " " + f"{'C-xling':>9}" if indic_real else ""
    print("\n" + "=" * (78 + len(xl)))
    print("PER-BLOCK PROBES   A lower=better | B lower=better | C-unseen higher=better")
    print("=" * (78 + len(xl)))
    print(f"{'block':>6} {'A lang':>8} {'B corpus':>9} {'C seen':>8} "
          f"{'C unseen':>9} {'C gap':>8}" + xl)
    print("-" * (78 + len(xl)))

    rows = []
    for b in blocks:
        a = probe_a(indic, b)
        bb = probe_b(en, indic, b, np.random.default_rng(SEED))
        c = probe_c(en, b, attack_of)
        r = {"block": str(b), "a_acc": a["acc"], "a_chance": a["chance"],
             "b_acc": bb["acc"], "b_chance": bb["chance"],
             "c_seen": c["auc_seen"], "c_unseen": c["auc_unseen"],
             "c_gap": c["gap"]}
        line = (f"{str(b):>6} {a['acc']*100:>7.2f}% {bb['acc']*100:>8.2f}% "
                f"{c['auc_seen']:>8.4f} {c['auc_unseen']:>9.4f} {c['gap']:>+8.4f}")
        if indic_real:
            x = probe_c_crosslingual(en, indic_real, indic, b,
                                     np.random.default_rng(SEED), args.n_cell)
            r["c_xling"] = x["auc_in"]
            r["c_xling_en"] = x["auc_en"]
            line += f" {x['auc_in']:>9.4f}"
        rows.append(r)
        print(line, flush=True)

    print("-" * (78 + len(xl)))
    print(f"{'chance':>6} {rows[0]['a_chance']*100:>7.2f}% "
          f"{rows[0]['b_chance']*100:>8.2f}% {0.5:>8.4f} {0.5:>9.4f}")
    return rows, en, indic, indic_real, args


# ===========================================================================
# Reporting
# ===========================================================================
def report(rows, en, indic, indic_real, args):
    """Rank the blocks, name the pick, and write the table to disk.

    The ranking is deliberately a rank-sum rather than a weighted score: A and
    B are accuracies, C is an AUC, and inventing an exchange rate between them
    would be a fabricated number dressed as a method. Rank-sum says only "this
    block is better on more axes", which is all the data supports.
    """
    base = next((r for r in rows if r["block"] == "ln"), None)

    def rank(key, lower_better):
        order = sorted(rows, key=lambda r: r[key], reverse=not lower_better)
        return {r["block"]: i for i, r in enumerate(order)}

    ra, rb = rank("a_acc", True), rank("b_acc", True)
    rc = rank("c_unseen", False)
    for r in rows:
        r["ranksum"] = ra[r["block"]] + rb[r["block"]] + rc[r["block"]]

    ranked = sorted(rows, key=lambda r: (r["ranksum"], -r["c_unseen"]))

    print("\n" + "=" * 78)
    print("RANKING   rank-sum over (A asc, B asc, C-unseen desc); lower is better")
    print("=" * 78)
    print(f"{'block':>6} {'ranksum':>8} {'A lang':>8} {'B corpus':>9} {'C unseen':>9}")
    for r in ranked[:8]:
        mark = "  <- v1 baseline" if r["block"] == "ln" else ""
        print(f"{r['block']:>6} {r['ranksum']:>8} {r['a_acc']*100:>7.2f}% "
              f"{r['b_acc']*100:>8.2f}% {r['c_unseen']:>9.4f}{mark}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if base is None:
        print("  No 'ln' block in this sweep, so there is no v1 baseline to")
        print("  compare against. Re-extract with --layers all,ln.")
    else:
        print(f"  v1 baseline ('ln' = last_hidden_state, what realtime/frontend.py")
        print(f"  serves today):  A {base['a_acc']*100:.2f}%   "
              f"B {base['b_acc']*100:.2f}%   C-unseen {base['c_unseen']:.4f}")
        w = ranked[0]
        if w["block"] == "ln":
            print("\n  The v1 representation ranks FIRST. No layer in this sweep beats")
            print("  it on the three axes together. That is a real result and it does")
            print("  NOT mean the S4 change failed -- pooling, not layer choice, is")
            print("  what docs/PROBE_RESULTS.md identified as the mechanism. Compare")
            print("  the mean-only and mean+std variants before concluding anything.")
        else:
            print(f"\n  Best block: {w['block']}")
            print(f"    A language  {w['a_acc']*100:6.2f}%  "
                  f"({(w['a_acc']-base['a_acc'])*100:+.2f} pts vs v1)")
            print(f"    B corpus    {w['b_acc']*100:6.2f}%  "
                  f"({(w['b_acc']-base['b_acc'])*100:+.2f} pts vs v1)")
            print(f"    C unseen    {w['c_unseen']:6.4f}   "
                  f"({w['c_unseen']-base['c_unseen']:+.4f} vs v1)")
            print(f"\n  Candidate v4 front-end:  --layers {w['block']} "
                  f"--pool {indic.meta['pool']}")

    print("\n  SCOPE. Probe C here is leave-attack-out inside ASVspoof, not the")
    print("  locked cross-lingual transfer probe. It says which block keeps spoof")
    print("  signal across an unseen GENERATOR, not across a language. Do not")
    print("  quote it against AUC 0.5196. The cross-lingual column needs Indic")
    print("  bonafide audio (S2); pass --indic-bonafide once it exists.")

    # ---- write the table -------------------------------------------------
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["block", "a_acc", "b_acc", "c_seen", "c_unseen", "c_gap", "ranksum"]
    if "c_xling" in rows[0]:
        cols.insert(-1, "c_xling")

    with open(out.rsplit(".", 1)[0] + ".csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Per-layer probe sweep\n\n")
        fh.write(f"`{args.en_root}` (n={len(en.X)}) + `{args.indic_fake_root}` "
                 f"(n={len(indic.X)}), pool `{indic.meta['pool']}`, "
                 f"{indic.P} dims per block.\n\n")
        fh.write("A = 5-way language within `data/indic_spoof` (chance "
                 f"{rows[0]['a_chance']*100:.2f}%), lower is better. "
                 "B = ASVspoof-spoof vs Indic-spoof, label held constant "
                 f"(chance {rows[0]['b_chance']*100:.2f}%), lower is better. "
                 "C = fit bonafide vs A01-A04, test vs A05+A06; higher unseen "
                 "is better.\n\n")
        fh.write("**Probe C is leave-attack-out inside ASVspoof, NOT the locked "
                 "cross-lingual probe. Do not quote it against AUC 0.5196.**\n\n")
        fh.write("| block | A lang | B corpus | C seen | C unseen | C gap | rank |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            nm = f"`{r['block']}`" + (" (v1)" if r["block"] == "ln" else "")
            fh.write(f"| {nm} | {r['a_acc']*100:.2f}% | {r['b_acc']*100:.2f}% | "
                     f"{r['c_seen']:.4f} | {r['c_unseen']:.4f} | "
                     f"{r['c_gap']:+.4f} | {r['ranksum']} |\n")
    print(f"\n  wrote {args.out} and {args.out.rsplit('.', 1)[0]}.csv")


if __name__ == "__main__":
    report(*main())
