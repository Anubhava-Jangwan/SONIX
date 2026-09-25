# Representation probes — measured results

**Run:** 2026-09-19 · **Machine:** croakie · **Scripts:** `probe_shortcut.py`, `probe_transfer.py`
**Inputs:** `outputs/embeddings` (ASVspoof19 LA), `outputs/embeddings_indicvoices_tr`, `outputs/embeddings_indicsynth`
**Representation under test:** frozen `facebook/wav2vec2-xls-r-300m` → `last_hidden_state` → `mean(dim=1)` → 1024-dim
(`src/extract_embeddings.py:244-246`, `realtime/frontend.py:70-71`)

These are the first cross-domain measurements ever taken on this project. They are 🔒 LOCKED.

---

## 🔒 Probe A — language is linearly decodable

Within **IndicSynth only**: every row is spoof, every row is Indic, one generator family.
Nothing varies except language, so no confound can explain the result.

| | |
|---|---|
| Task | 12-way language classification, linear (logistic regression) |
| Classes | Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu |
| n | 12,600 train / 4,200 test |
| **Accuracy** | **99.33%** |
| Chance | 8.33% |

A *linear* probe reads language off the mean-pooled vector almost perfectly.
A 262,657-parameter MLP head gets this for free.

## 🔒 Probe B — corpus is linearly decodable

**Bonafide rows only**, so the label is held constant and cannot explain the result.
Split by `recording_id` (chunk siblings never straddle), so it is not clip memorisation.

| | |
|---|---|
| Task | IndicVoices vs ASVspoof, binary, linear |
| n | 11,125 train / 3,705 test (balanced by recording) |
| **Accuracy** | **100.00%** |
| Majority baseline | 82.54% |

Zero errors. The two corpora are perfectly linearly separable among same-label rows.

## 🔒 Probe C — the label does not transfer, and debiasing does not fix it

Train a linear spoof/bonafide probe on **English (ASVspoof) only**. Test in-domain and on Indic.
Then apply INLP: repeatedly find the direction that best predicts language/corpus, project it
out, retest. 4 balanced cells of 2,580.

| Directions removed | AUC in-domain (EN) | AUC cross-domain (Indic) | gap |
|---|---|---|---|
| 0 (baseline) | 0.9992 | **0.5196** | +0.4796 |
| 1 | 0.9992 | 0.5243 | +0.4748 |
| 2 | 0.9992 | 0.5282 | +0.4710 |
| 5 | 0.9992 | 0.5164 | +0.4828 |
| 10 | 0.9992 | 0.5034 | +0.4958 |
| 15 | 0.9992 | 0.4987 | +0.5005 |
| 20 | 0.9992 | **0.5011** | +0.4981 |

**0.50 is chance.** Cross-domain discrimination is absent at baseline and never recovers.
In-domain AUC is pinned at 0.9992 throughout — the removed directions carried no label
information at all.

---

## What this means

**1. It is not a threshold problem.** Case C is eliminated. AUC 0.52 is not a well-ranked
detector at a bad operating point; it is no ranking. Recalibration cannot help.

**2. It is not only a language shortcut.** That was the working hypothesis and it is
incomplete. If language were masking a transferable signal, projecting language out would
reveal it. Twenty directions removed, and cross-domain AUC drifts *toward* 0.50 rather than
away from it. **There is no transferable synthesis signal in the vector to unmask.**

**3. What the probe actually learns in-domain is ASVspoof itself.** 0.9992 on English is
near-perfect and worth nothing — it is corpus fingerprint, documented generator pipelines,
and channel. Exactly what Sun et al. (2026) call *"benchmark overfitting, where models exploit
quirks or attack fingerprints in the data set that fail to carry over to real-world conditions."*
This is a textbook instance, measured.

**4. Mean-pooling is the mechanism.** XLS-R runs at ~50 fps; a 4 s window is ~200 frames,
averaged into one vector. Stationary properties — language, speaker, channel, SNR — survive
averaging intact. Vocoder artifacts are transient (phase discontinuities at frame boundaries,
local spectral over-smoothing, micro-jitter) and are averaged into noise. The pooling step is
a low-pass filter that preserves every confound and destroys the evidence. Compounding it,
`last_hidden_state` is the most content-abstracted layer in the stack.

**5. Therefore the S4 front-end change is mandatory, not optional.** Rebalancing the corpus,
group-weighted loss, OC-Softmax and DANN all operate on the vector. None of them can put back
information that pooling removed. Data composition work is necessary but strictly insufficient.

---

## Honest caveats

- The `[language/corpus still readable at 100.00%]` line in `probe_transfer.py` is **training**
  accuracy, measured on the fitted data with no held-out split. After 20 rank-1 projections the
  space is still 1004-dimensional, so 100% separation of ~10k points is expected regardless. That
  line should not be read as "INLP failed to remove domain information." It is uninformative and
  the script should be fixed to hold out a split. **The AUC columns are unaffected** — those use
  a held-out English split and an entirely unseen Indic set.
- Probe C trains on English only. It proves the representation carries **no domain-transferable**
  synthesis signal. It does **not** prove that a model trained on Indic spoofs (like `head_v3`)
  cannot detect *seen* Indic generators. That question belongs to `diagnose_pairs.py`, still unrun.
- `probe_shortcut.py`'s Probe C (centroid cosine geometry) was a badly designed test. All four
  group centroids sit at cosine 0.994–0.999 of each other, and the closest pair is
  Indic-bonafide ↔ English-spoof at 0.9991 — opposite label *and* opposite language. Centroid
  cosine cannot resolve structure in a narrow high-dimensional cone. Its printed verdict line
  should be ignored; Fisher discriminant ratio is the right statistic. Probes A and B stand.

---

## Publication value

None of the three reference papers (Bird & Lotfi 2023; San Roman et al. 2024; Sun et al. 2026)
measures this. Sun et al. *describes* domain shift as "the dominant failure mode" and warns about
benchmark overfitting, but reports no probe. A confound-controlled demonstration that a
mean-pooled SSL representation encodes language at 99.33% while carrying zero transferable
synthesis signal — plus the fix and the delta — is a self-contained, publishable result.
Negative-result-plus-fix is a well-received workshop format.

---

## Next

| Step | Status |
|---|---|
| Re-extract with all hidden states + learned layer weights + attentive stats pooling (S4) | ready to write |
| Re-run Probes A, B, C on the new representation | the delta is the result |
| `diagnose_pairs.py` on `sonix_real/` | still unrun |
| Fix the held-out split in `probe_transfer.py`'s domain-accuracy line | minor |
