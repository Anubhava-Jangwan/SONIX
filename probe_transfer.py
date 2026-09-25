#!/usr/bin/env python3
"""
probe_transfer.py  --  SONIX / SIH26104

Reproduces the ACTUAL bug as one number, then tests whether debiasing fixes it.

Companion to probe_shortcut.py, which already established (2026-09-19):
    Probe A  language, 12-way, within IndicSynth only   99.33%  (chance 8.33%)
    Probe B  corpus, bonafide rows only, recording-split 100.00% (chance 82.54%)

Those say language and corpus ARE linearly decodable from the current
mean-pooled XLS-R vectors. This script asks the follow-up question that
decides the architecture:

    STEP 1  Train a linear spoof/bonafide probe on ENGLISH only (ASVspoof).
            Test in-domain (English) and cross-domain (Indic).
            The gap IS head_robust_v2 / head_v3's failure, reduced to a linear
            model so there is nothing else to blame it on.

    STEP 2  INLP (iterative nullspace projection). Repeatedly find the linear
            direction that best predicts LANGUAGE/CORPUS, project it out of the
            embedding space, repeat. Then redo STEP 1 on the debiased space.

HOW TO READ THE RESULT
    cross-domain AUC RECOVERS as directions are removed
        -> the label signal survives without language. Debiasing + rebalancing
           is enough. Keep the frozen XLS-R front-end, change S5/S6 only.

    cross-domain AUC stays flat or DROPS
        -> label information is entangled with language in this representation,
           or was never there to begin with (mean-pooling destroyed it).
           No amount of reweighting saves it. The S4 front-end change
           (all hidden states + attentive stats pooling) is mandatory.

    in-domain AUC collapsing too
        -> the directions being removed carry the label as well. Same verdict
           as above, and a stronger one.

Run (Linux VM shell, numpy + scikit-learn):
    python3 probe_transfer.py

Needs, relative to the repo root:
    outputs/embeddings/train                 ASVspoof19 LA  (English, both labels)
    outputs/embeddings_indicvoices_tr/train  IndicVoices    (Indic, bonafide)
    outputs/embeddings_indicsynth/train      IndicSynth     (Indic, spoof)
"""

import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

N_PER_CELL = 2580      # size of each of the 4 group cells; ASVspoof bonafide caps it
N_DIRECTIONS = 20      # INLP iterations
REPORT_AT = (1, 2, 5, 10, 15, 20)


def load(rel, split="train"):
    """Concatenate every shard in one embedding root/split -> (X float32, y int)."""
    d = os.path.join(ROOT, rel, split)
    Xs, ys = [], []
    for npy in sorted(glob.glob(os.path.join(d, "*.npy"))):
        if npy.endswith(".labels.npy"):
            continue
        Xs.append(np.load(npy))
        ys.append(np.load(npy[:-4] + ".labels.npy"))
    if not Xs:
        raise SystemExit(f"no shards found in {d}")
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def subsample(X, n):
    if len(X) <= n:
        return X
    return X[rng.choice(len(X), size=n, replace=False)]


def halves(A, B):
    """Split two class arrays 50/50 into train and test, keeping labels aligned."""
    ia, ib = len(A) // 2, len(B) // 2
    Xtr = np.vstack([A[:ia], B[:ib]])
    ytr = np.r_[np.zeros(ia, int), np.ones(ib, int)]
    Xte = np.vstack([A[ia:], B[ib:]])
    yte = np.r_[np.zeros(len(A) - ia, int), np.ones(len(B) - ib, int)]
    return Xtr, ytr, Xte, yte


def evaluate(tag, project):
    """Train spoof/bonafide on English, report AUC in-domain and on Indic."""
    Xtr, ytr, Xen, yen = halves(project(EN_REAL), project(EN_FAKE))
    Xin = np.vstack([project(IN_REAL), project(IN_FAKE)])
    yin = np.r_[np.zeros(len(IN_REAL), int), np.ones(len(IN_FAKE), int)]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(sc.transform(Xtr), ytr)

    auc_en = roc_auc_score(yen, clf.predict_proba(sc.transform(Xen))[:, 1])
    auc_in = roc_auc_score(yin, clf.predict_proba(sc.transform(Xin))[:, 1])
    print(f"    {tag:34s}  AUC(EN) = {auc_en:.4f}   AUC(IN) = {auc_in:.4f}"
          f"   gap = {auc_en - auc_in:+.4f}")
    return auc_en, auc_in


print("loading embeddings ...")
Xa, ya = load("outputs/embeddings")
Xiv, _ = load("outputs/embeddings_indicvoices_tr")
Xis, _ = load("outputs/embeddings_indicsynth")

EN_REAL = subsample(Xa[ya == 0], N_PER_CELL)
EN_FAKE = subsample(Xa[ya == 1], N_PER_CELL)
IN_REAL = subsample(Xiv, N_PER_CELL)
IN_FAKE = subsample(Xis, N_PER_CELL)
DIM = EN_REAL.shape[1]

print(f"  cells: EN real {len(EN_REAL)} | EN fake {len(EN_FAKE)} | "
      f"IN real {len(IN_REAL)} | IN fake {len(IN_FAKE)} | dim {DIM}")

print("\n" + "=" * 86)
print("STEP 1  --  baseline: train spoof/bonafide on English, test on Indic")
print("=" * 86)
base_en, base_in = evaluate("raw mean-pooled XLS-R", lambda X: X)

print("\n" + "=" * 86)
print("STEP 2  --  INLP: remove language/corpus directions, retest after each")
print("=" * 86)

DOMAIN_X = np.vstack([IN_REAL, IN_FAKE, EN_REAL, EN_FAKE])
DOMAIN_Y = np.r_[
    np.ones(len(IN_REAL) + len(IN_FAKE), int),    # Indic
    np.zeros(len(EN_REAL) + len(EN_FAKE), int),   # English
]

P = np.eye(DIM, dtype=np.float32)
for step in range(1, N_DIRECTIONS + 1):
    Xp = DOMAIN_X @ P
    sc = StandardScaler().fit(Xp)
    clf = LogisticRegression(max_iter=400, n_jobs=-1).fit(sc.transform(Xp), DOMAIN_Y)
    domain_acc = clf.score(sc.transform(Xp), DOMAIN_Y)

    w = (clf.coef_[0] / sc.scale_).astype(np.float32)
    w /= np.linalg.norm(w) + 1e-9
    P = (P @ (np.eye(DIM, dtype=np.float32) - np.outer(w, w))).astype(np.float32)

    if step in REPORT_AT:
        print(f"\n  after {step:2d} direction(s) removed"
              f"   [language/corpus still readable at {domain_acc * 100:.2f}%]")
        evaluate("debiased", lambda X, P=P: X @ P)

print("\n" + "=" * 86)
print("VERDICT")
print("=" * 86)
print(f"  baseline cross-domain AUC(IN) = {base_in:.4f}")
print("  Compare against the final debiased AUC(IN) above.")
print("    recovers  -> debiasing is enough, keep the front-end")
print("    flat/down -> S4 front-end change is mandatory")
print()
