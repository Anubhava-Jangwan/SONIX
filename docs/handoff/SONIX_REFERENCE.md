# SONIX — Complete Project Reference

**Hand this whole file to your AI assistant before asking it for help.** It contains everything it needs to know about the project, the current problem, and your job.

---

## 0. How to use this file (instructions for the AI assistant)

You are helping a student team at Smart India Hackathon 2026. Read this entire document first. Key things to respect:

- **Never invent a number.** If a metric is not written in this file, say it is unknown. Do not estimate EERs.
- The metric is **EER**, not accuracy. The data is ~90% fake, so accuracy is meaningless here.
- **Never suggest training on the eval set, the In-the-Wild set, or the 2021-DF set.** Those are the honest test sets. Training on them invalidates the only numbers that prove generalisation.
- Sanity ranges: low single-digit EER = working. ~40% EER = a label bug. ~0% EER = a data leak. React accordingly.
- The team is under a hard deadline (Stage 2 PoC, 1–2 Sept 2026). Prefer small, safe, high-impact changes over rewrites.

---

## 1. What SONIX is

- **Competition:** Smart India Hackathon 2026, Problem Statement **SIH26104** (AICTE Cyber Security Cell).
- **Title:** AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks.
- **Team:** Sonix, Team ID SIH-UPES-2026-T302. **Stage 2 PoC round: 1–2 Sept 2026 at UPES.**
- **Repo:** github.com/Anubhava-Jangwan/SONIX · local path `C:\Users\yugal\OneDrive\Desktop\Sonix`

**The pitch.** Caller ID is spoofable. Recognising the voice is exactly what a cloning attack exploits. Fraud systems only see the transaction afterwards, so the victim gets no signal during the call. SONIX supplies that signal **while the call is live** — prevention, not post-incident forensics.

**Critical design point:** SONIX **never blocks a call.** It raises a Green / Amber / Red risk band and flags for human verification. A human always makes the final decision. Thresholds are configurable per organisation. This is a deliberate safety property and a selling point, not a limitation.

---

## 2. How the system works (pipeline)

```
audio -> 4-second windows at 0.5s hop
      -> wav2vec2 XLS-R 300M (FROZEN self-supervised front-end)
      -> mean-pool over time -> 1024-dim embedding
      -> small trained MLP head (262,657 params)
      -> sigmoid -> spoof probability 0..1  (higher = more likely FAKE)
      -> 5-window moving average + hysteresis
      -> Green / Amber / Red band + recommended action
```

- Only the **head** is trained. The wav2vec2 front-end is frozen and never updated.
- Head architecture: `Linear(1024,256) -> ReLU -> Dropout(0.3) -> Linear(256,1)`.
- Loss: `BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos` (handles the ~9:1 spoof:bonafide imbalance).
- Embeddings are **precomputed and cached to disk**, so head training takes minutes, not hours. If training takes hours, something is re-extracting features — stop and find it.
- Input standardisation: embeddings are z-scored using **train-split statistics only**; `mu`/`sd` are stored inside the checkpoint so eval and the demo apply the identical transform.

---

## 3. Current results (these are the only real numbers — do not invent others)

| What | Number | Notes |
|---|---|---|
| Baseline `head.pt`, ASVspoof-2019 LA **eval** EER | **1.49%** | unseen attacks A07–A19, threshold 0.0951 |
| Baseline `head.pt`, dev EER | 0.16% | dev shares attacks with train — near-0 is expected, not a leak |
| Augmented `head_aug.pt`, dev EER | **0.24%** | best at epoch 11, early-stopped. Dev here is clean+codec, so NOT comparable to the baseline's clean dev |
| Real-world mic recordings | **~50–60% of genuine clips wrongly flagged** | this is the problem we are fixing |
| 2021-DF EER | not yet measured | embeddings extracted, scoring pending |
| In-the-Wild EER | not yet measured | embeddings extracted, scoring pending |
| Codec 2×2 EERs | not yet measured | eval codec'ing in progress |

**Thresholds.** From the clean eval score distribution (strongly bimodal near 0 and 1) the data-driven thresholds are **amber 0.10, red 0.90**, and these are now the UI defaults. **These are known to be too trigger-happy on real-world audio** and are being recalibrated on real recordings.

---

## 4. THE CURRENT PROBLEM (most important section)

**Symptom.** Roughly 50–60% of genuine, real mic recordings are flagged Amber or Red by the baseline model. Clips with long quiet stretches ("minimal voice") are worst.

**Diagnosis — domain shift, not a bug.** The model was trained only on ASVspoof-2019 LA, whose bonafide audio is clean studio recording (one microphone type, quiet room, consistent channel). Real phone/laptop recordings differ in mic, room acoustics, background noise and compression. The model has never seen that distribution of *real* audio, so genuine speech drifts toward the "spoof" side of its decision boundary.

Two stacking causes:

1. **Silence / low-energy windows.** Near-silent 4-second chunks have no calibrated behaviour; the model often scores them high. wav2vec2's feature extractor normalises each input, which amplifies noise in quiet windows.
2. **Threshold mismatch.** Amber 0.10 was derived from a clean, bimodal score distribution. Out-of-domain real audio lands in the mushy middle, so 0.10 fires constantly.

**This is a known, field-wide problem**, which is exactly why the In-the-Wild benchmark exists. It is not a defect unique to this team, and the PPT already anticipates it.

**The fix — three things, in impact order:**

1. **More diverse real (bonafide) audio in training** plus augmentation (RawBoost, codec). Biggest lever.
2. **VAD / energy gate** so near-silent windows are skipped rather than scored.
3. **Recalibrate amber/red** on actual real vs cloned recordings.

**What will NOT fix it:** swapping in a fancier classifier (AASIST, RawNet2) while training on the same narrow data. This is a *data* problem.

**Diagnostic tool.** `demo/diagnose.py` prints, per 4-second window, the energy in dBFS and the spoof score for both models, plus a summary splitting speech windows from silence windows.

```
cd demo
python diagnose.py "C:\path\to\clip.wav"
```

If silence windows score high and speech windows score low, the VAD gate is the fix. If speech windows themselves score high, it is pure domain shift, needing data plus recalibration.

---

## 5. Repository layout

```
Sonix/
  src/
    verify_protocol.py      # dataset/protocol sanity gate - run first
    extract_embeddings.py   # audio -> cached 1024-d embeddings (train/dev/eval/itw, or --audio-dir)
    train.py                # trains the MLP head; --extra-emb-root adds augmented data
    eval.py                 # scores a split, saves scores for metrics
    score_file.py           # (source copy)
  demo/                     # the Streamlit demo - THIS is what runs on stage
    app.py                  # UI: Baseline / Augmented tabs, thresholds, live timeline
    score_file.py           # real-model bridge: score_file() / score_stream()
    model.py                # re-exports score_file/score_stream
    model_adapter.py        # UI <-> model adapter, passes the chosen checkpoint
    risk.py                 # moving average + hysteresis -> Green/Amber/Red
    streaming.py            # window iteration + mock score generator
    windowing.py            # audio loading + 4s window slicing
    diagnose.py             # per-window energy + score diagnostic
  uidemo/SONIX_Suryansh_Demo_UI_v9/   # mirrored copy of demo/ (submodule), kept identical
  make_codec.py             # G.711 telephony codec simulation
  data/asvspoof19_la/       # dataset (gitignored)
  outputs/                  # embeddings, models, scores (gitignored)
    embeddings/{train,dev,eval,itw}
    embeddings_g711/{train,dev}
    models/head.pt, head_aug.pt
```

**Demo internals worth knowing:**

- The frozen wav2vec2 front-end is loaded **once** and shared; each checkpoint only swaps in its small head plus `mu`/`sd`. So baseline and augmented can run back-to-back without reloading 300M params.
- Checkpoint paths resolve by **searching upward** for `outputs/models/<name>`, so the demo works whether launched from `demo/` or the repo root.
- `score_stream(wav, ckpt_path=...)` yields one score per window, in order, for the live UI. `score_file(wav, ckpt_path=...)` returns the whole list for offline/benchmark use.

---

## 6. Datasets

| Split | Count | Location |
|---|---|---|
| train | 25,380 | `data/asvspoof19_la` |
| dev | 24,844 | `data/asvspoof19_la` |
| eval | 71,237 | `data/asvspoof19_la` |

- Clean embeddings: `outputs/embeddings/{train,dev,eval,itw}` (254 train shards, 249 dev shards).
- G.711 codec embeddings: `outputs/embeddings_g711/{train,dev}` (254 / 249 shards).
- Augmented training set = 50,760 vectors (clean 25,380 + codec 25,380).
- ASVspoof-2021 DF: embeddings extracted in 4 parts, currently on an external SSD. **Needs the DF CM key file** to map trial IDs to bonafide/spoof — without it there is no EER.
- In-the-Wild: embeddings exist at `outputs/embeddings/itw`. **Test only.**
- Indic evaluation set: being built now from team recordings plus open Indic TTS.

---

## 7. Key commands

```powershell
# environment
cd C:\Users\yugal\OneDrive\Desktop\Sonix
.\.venv\Scripts\Activate.ps1

# sanity gate
python src\verify_protocol.py

# embeddings
python src\extract_embeddings.py --split train --batch 8        # also dev, eval
python src\extract_embeddings.py --split X --audio-dir "<folder>" --out outputs\embeddings_X --batch 8

# train baseline head
python src\train.py

# train augmented (codec-robust) head
python src\train.py --emb-root outputs\embeddings --extra-emb-root outputs\embeddings_g711 --out outputs\models\head_aug.pt

# evaluate
python src\eval.py --split eval --model-ckpt outputs\models\head.pt

# codec simulation
python make_codec.py --split eval

# the demo
cd demo
streamlit run app.py

# single-clip scoring / diagnosis
python score_file.py clip.wav --ckpt outputs\models\head_aug.pt
python diagnose.py clip.wav
```

---

## 8. The plan (as of 31 Aug) — three lanes in parallel

**LANE 1 — FIX THE MODEL** (Yugal, Akshat, Yukti)
VAD/energy gate; collect 30–40 diverse real recordings plus 15–20 cloned; RawBoost augmentation; retrain `head_robust.pt` on clean + G.711 + RawBoost + real audio; recalibrate thresholds on real clips.

**LANE 2 — HONEST NUMBERS** (Navya, Anubhav, Yukti)
2021-DF EER; In-the-Wild EER (test only); codec 2×2 = {baseline, augmented} × {clean eval, G.711 eval}; `metrics.py` as single source of truth; temperature-scaling calibration.

**LANE 3 — A DEMO THAT CANNOT FAIL** (Suryansh, Anubhav, Akshat)
Replay mode that streams **precomputed** scores for validated clips (so the stage demo never depends on live inference on unpredictable audio); clip-library dropdown; opening pitch; jury Q&A; backup video.

**Timeline.** Tonight: recording, codec'ing, DF/ITW scoring, VAD + RawBoost code, replay mode, metrics. Overnight: embedding extraction on the two RTX 4060s. Tomorrow AM: retrain robust head, set real thresholds, load score JSONs, rehearse.

---

## 9. Team, machines, and assignments

| Person | Machine | Owns |
|---|---|---|
| **Yugal** | RTX 3050 | ML lead. VAD gate, `make_rawboost.py`, retrain robust head, OC-Softmax (stretch), precompute demo scores, integrate thresholds |
| **Akshat** | RTX 4060 | 30–40 real clips plus 15–20 cloned (Hindi + English, many devices), **real/ and fake/ in separate folders**, manifest, embedding extraction |
| **Navya** | RTX 4060 | 2021-DF EER (blocked on CM key), In-the-Wild EER (test only), then RawBoost extraction batch |
| **Anubhav** | GTX 1650 | Codec eval extraction, codec 2×2 numbers, opening pitch, jury Q&A |
| **Suryansh** | No GPU | Demo replay mode, clip library, model tabs, backup video |
| **Yukti** | No GPU | `metrics.py`, temperature calibration, real-world amber/red thresholds, results table |

**Why real/ and fake/ must be separate folders:** it gives us labels for free (everything in `real/` is bonafide = 0, everything in `fake/` is spoof = 1) with zero manual labelling and zero risk of a label bug.

---

## 10. What the PPT promises vs what is built

| PPT claim | Status |
|---|---|
| wav2vec2 / WavLM frozen front-end | **BUILT** (wav2vec2 XLS-R 300M) |
| Classifier head | **BUILT** — MLP baseline (PPT named AASIST) |
| AASIST head | **PLANNED** — needs frame-level features; the current pipeline mean-pools them away, so it is a pipeline rebuild, not a tweak. Finale item |
| RawNet2 comparison baseline | **PLANNED** |
| OC-Softmax one-class objective | **PLANNED** — cheap-ish (loss swap on the head), stretch goal |
| RawBoost augmentation | **IN PROGRESS** — being added now |
| Codec G.711 / Opus / AMR | **G.711 done**, Opus/AMR planned |
| Temperature-scaling calibration | **IN PROGRESS** (Yukti) |
| ASVspoof 2019 LA | **DONE** — 1.49% eval EER |
| ASVspoof 2021 DF | **IN PROGRESS** |
| In-the-Wild | **IN PROGRESS** (test only) |
| Indic evaluation set | **IN PROGRESS** — being built from team recordings |
| FastAPI / WebSocket / React / Docker | **Out of PoC scope** — the feasibility slide already calls this product packaging, not the unsolved problem |

**The honest framing that holds up.** The PPT methodology slide states: *"Reproduce a published baseline first, then add augmentation and one-class training, then measure cross-dataset and Indic performance, then convert to windowed streaming."* The team is legitimately at the baseline + augmentation stage, with windowed streaming already working. An MLP head on frozen SSL embeddings is a valid published-style baseline. AASIST, RawNet2 and OC-Softmax are the **stated next stages**, not omissions.

---

## 11. Hard rules

1. **Never invent a number.** Only quote metrics that were actually measured.
2. **EER, not accuracy.** The data is ~90% spoof.
3. **Never train on eval, In-the-Wild, or 2021-DF.** They are the honest test.
4. Sanity check: single-digit EER = working; ~40% = label bug; ~0% = leak.
5. The system **never blocks a call** — it flags for a human.
6. Keep `demo/` and `uidemo/SONIX_Suryansh_Demo_UI_v9/` byte-identical.
7. Do not re-extract embeddings you already have — training on cached embeddings takes minutes.

---

## 12. Glossary

- **EER (Equal Error Rate):** the single error number where false accepts equal false rejects. Lower is better. 1.49% is strong; 50% is a coin flip.
- **bonafide:** a real human voice. **spoof:** an AI-cloned/synthetic voice. Label 1 = spoof.
- **embedding:** the 1024-number fingerprint of a 4-second audio chunk produced by the frozen front-end.
- **head:** the small trainable model on top of the frozen front-end.
- **front-end:** wav2vec2 XLS-R 300M, frozen — never trained.
- **VAD:** Voice Activity Detection — is anyone actually speaking in this chunk?
- **augmentation:** altered copies of training audio (phone-compressed, noisy, different channel) so the model sees variety.
- **RawBoost:** an augmentation method that simulates convolutive, impulsive and coloured additive noise — that is, different mics and channels.
- **OC-Softmax:** a one-class training objective that learns a compact region for bonafide, improving rejection of *unseen* attacks.
- **calibration:** making the 0–1 output a real probability so thresholds mean something.
- **domain shift:** test audio differs from training audio. This is the team's core current problem.
- **hysteresis:** requiring several consecutive agreeing windows before changing the risk band, so it does not flicker.

---

*Reference written 31 Aug 2026 for SONIX / SIH26104. If a fact is not in this document, it is not established — ask the team rather than assuming.*
