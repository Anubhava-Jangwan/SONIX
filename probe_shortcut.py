#!/usr/bin/env python3
"""
probe_shortcut.py -- SONIX / SIH26104

Measures whether the CURRENT embedding space (frozen XLS-R, last_hidden_state,
mean-pooled) encodes language/corpus more strongly than it encodes synthesis.

Three probes, each confound-controlled:

  A) LANGUAGE, within one corpus and one label.
     IndicSynth only: every row is spoof, Indic, same generator family.
     Nothing varies except language. If a LINEAR probe reads language off
     these vectors, language is trivially available to a 262k head.

  B) CORPUS, within one label.
     IndicVoices bonafide vs ASVspoof bonafide. Both label=0, so the label
     cannot explain the result. Split by RECORDING ID (chunk siblings never
     straddle), so it is not memorising clips.

  C) GEOMETRY.
     Cosine distance between the four group centroids. If the space is
     organised by language, the two Indic centroids sit closer to each other
     than either sits to its same-label English counterpart.

Run:  python3 probe_shortcut.py
"""
import glob, os, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_root(rel, split="train"):
    """Return (X float32, y int, files list) for one embedding root/split."""
    d = os.path.join(ROOT, rel, split)
    Xs, ys, fs = [], [], []
    for npy in sorted(glob.glob(os.path.join(d, "*.npy"))):
        if npy.endswith(".labels.npy"):
            continue
        stem = npy[:-4]
        Xs.append(np.load(npy))
        ys.append(np.load(stem + ".labels.npy"))
        with open(stem + ".files.txt") as fh:
            fs += [l.strip() for l in fh if l.strip()]
    if not Xs:
        raise SystemExit(f"no shards in {d}")
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    assert len(fs) == len(X), f"{rel}: {len(fs)} names vs {len(X)} rows"
    return X, y, fs

def fit_probe(X, y, groups=None, seed=0, name=""):
    """Linear probe. If groups given, split by group so siblings don't leak."""
    if groups is not None:
        uniq = np.unique(groups)
        tr_g, te_g = train_test_split(uniq, test_size=0.25, random_state=seed)
        tr = np.isin(groups, tr_g); te = np.isin(groups, te_g)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    acc = clf.score(sc.transform(Xte), yte)
    chance = np.bincount(yte).max() / len(yte)
    print(f"  {name}: accuracy = {acc*100:.2f}%   (majority-class baseline "
          f"{chance*100:.2f}%, n_train={len(ytr)}, n_test={len(yte)})")
    return acc, chance

# ---------------------------------------------------------------- PROBE A
print("\n" + "="*70)
print("PROBE A -- LANGUAGE, within IndicSynth only")
print("  all rows spoof, all Indic, one generator family.")
print("="*70)
X, y, f = load_root("outputs/embeddings_indicsynth")
langs = [re.match(r"indicsynth_([A-Za-z]+)_", n).group(1) for n in f]
names = sorted(set(langs))
lab = np.array([names.index(l) for l in langs])
print(f"  {len(names)} languages: {', '.join(names)}")
print(f"  label sanity: spoof={int((y==1).sum())} bonafide={int((y==0).sum())}")
accA, chA = fit_probe(X, lab, name="language (12-way)")

# ---------------------------------------------------------------- PROBE B
print("\n" + "="*70)
print("PROBE B -- CORPUS, bonafide rows only (label held constant)")
print("="*70)
Xi, yi, fi = load_root("outputs/embeddings_indicvoices_tr")
Xa, ya, fa = load_root("outputs/embeddings")
Xi, fi = Xi[yi == 0], [n for n, v in zip(fi, yi) if v == 0]
Xa, fa = Xa[ya == 0], [n for n, v in zip(fa, ya) if v == 0]
print(f"  IndicVoices bonafide = {len(Xi)}   ASVspoof bonafide = {len(Xa)}")
# recording id: strip _chunk_N so siblings stay together
rec_i = np.array([re.sub(r"_chunk_\d+.*$", "", n) for n in fi])
rec_a = np.array([re.sub(r"\..*$", "", n) for n in fa])
# balance to the smaller side, by recording
rng = np.random.default_rng(0)
keep = rng.choice(np.unique(rec_i), size=min(len(np.unique(rec_i)),
                  len(np.unique(rec_a))), replace=False)
m = np.isin(rec_i, keep)
Xb = np.concatenate([Xi[m], Xa])
yb = np.concatenate([np.zeros(m.sum(), int), np.ones(len(Xa), int)])
gb = np.concatenate([rec_i[m], rec_a])
print(f"  balanced: IndicVoices {int(m.sum())} vs ASVspoof {len(Xa)}")
accB, chB = fit_probe(Xb, yb, groups=gb, name="corpus (2-way)")

# ---------------------------------------------------------------- PROBE C
print("\n" + "="*70)
print("PROBE C -- GEOMETRY of the four group centroids")
print("="*70)
Xs, ys_, _ = load_root("outputs/embeddings")
groups = {
    "Indic  / bonafide": Xi,
    "Indic  / spoof   ": X,
    "Englsh / bonafide": Xa,
    "Englsh / spoof   ": Xs[ys_ == 1],
}
cent = {k: v.mean(0) for k, v in groups.items()}
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
keys = list(cent)
print("\n  cosine similarity between centroids (higher = closer):\n")
print("                      " + "".join(f"{k[:9]:>12s}" for k in keys))
for a in keys:
    print(f"  {a}" + "".join(f"{cos(cent[a],cent[b]):>12.4f}" for b in keys))
same_lang_diff_label = cos(cent["Indic  / bonafide"], cent["Indic  / spoof   "])
same_label_diff_lang = cos(cent["Indic  / bonafide"], cent["Englsh / bonafide"])
print(f"\n  Indic-real  <-> Indic-fake    (same language, DIFFERENT label): {same_lang_diff_label:.4f}")
print(f"  Indic-real  <-> English-real  (same label, DIFFERENT language): {same_label_diff_lang:.4f}")

print("\n" + "="*70)
print("VERDICT")
print("="*70)
print(f"  A language probe     : {accA*100:6.2f}%  (chance {chA*100:.2f}%)")
print(f"  B corpus probe       : {accB*100:6.2f}%  (chance {chB*100:.2f}%)")
if same_lang_diff_label > same_label_diff_lang:
    print("  C geometry           : space is organised by LANGUAGE, not by label.")
else:
    print("  C geometry           : space is organised by LABEL. Good.")
print()
