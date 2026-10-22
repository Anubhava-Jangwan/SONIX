# SONIX — Complete Project Context

**Paste this whole file into a new chat as the first message. It is self-contained.**

Smart India Hackathon 2026 · Problem Statement **SIH26104** · AICTE Cyber Security Cell
Team SONIX · Team ID **SIH-UPES-2026-T302** · UPES Dehradun
Theme: Blockchain & Cybersecurity · Category: Software
Compiled 19 September 2026.

---

## 0. READ THIS FIRST — the rules that govern every answer

1. **Never invent a number.** §5 is the ledger of what is measured. Anything not
   there is `[[PLACEHOLDER]]`. A wrong number in front of a jury is fatal; a blank
   is survivable.
2. **The metric is EER, not accuracy.** The benchmark data is ~90 % spoof, so a
   model that answers "fake" every time scores 90 % accuracy and is useless.
3. **Score runs 0–1 and HIGHER means the model thinks it is FAKE.**
4. **Never train on the held-out sets** (§4.3). Not once, not "just to see".
5. **Dev EER is not real-world performance.** See §6.1 — this is the single most
   important conceptual point in the project and the source of the current confusion.

---

## 1. What SONIX is

Real-time detection of AI-cloned voices **during a live call**. It scores the audio
stream continuously and surfaces Green / Amber / Red. **It never blocks a call — it
flags for a human.**

The problem statement demands four things, and the third is the one most teams miss:

- detect AI-generated/cloned speech **in real time**, not from a recording afterwards
- work on **"diverse Indian accents and dialects"** (stated explicitly in the PS)
- expose **"REST/gRPC APIs and SDKs"** for banking, contact centre and telecom
  integration — the PS says this **three separate times**
- be a **"reusable security layer"** — middleware other systems call, not an app

---

## 2. Architecture

```
live call audio (16 kHz mono, VoIP / telephony / mic)
  → consent gate           frames dropped in the state machine until approved
  → ring buffer            4.0 s window, 0.5 s hop → 87.5 % overlap
  → silence gate (VAD)     energy + zero-crossing rate; adaptive floor
  → wav2vec2 XLS-R 300M    FROZEN, 0 trainable params; mean-pool → 1024-dim
  → MLP head               TRAINED, 262,657 params
                           1024 → 256 → ReLU → Dropout(0.3) → 1 → sigmoid
  → sigmoid                P(synthetic), 0–1
  → 5-window moving average
  → hysteresis             3 of 5 windows must agree before the band switches
  → GREEN (<0.10) / AMBER / RED (>0.90)   thresholds operator-configurable
  → a human decides
```

**The design decision everything rests on — "frozen listener, swappable judge".**
300M frozen parameters listen; 262,657 trained parameters decide. A new cloning tool
appears → extract embeddings once → retrain the head in **minutes** → serve it from
the same endpoint alongside the old one. No redeployment, no downtime, no second GPU.
Seven heads are loaded and hot-swappable today, all sharing one resident front-end.

**Adaptive VAD floor:** `clip(90th-percentile frame RMS × 0.12, 0.0006, 0.01)`.
A fixed floor rejected quiet recordings wholesale.

**Class imbalance** handled with `pos_weight = n_neg / n_pos`.

**Recording-level holdout** (not row-level) — `<id>_chunk_N` siblings must not
straddle train/test.

### Serving
7 REST endpoints + WebSocket:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | health, mode, active calls, engine stats |
| GET | `/api/models` | which heads are loaded |
| POST | `/api/score-file` | submit audio; returns immediately, streams results |
| GET | `/api/telemetry?call_id=` | per-window scores, VAD decisions, ring-buffer stats |
| POST | `/api/approve` | approve the pairing/consent code |
| POST | `/api/end-call` | finalise and archive |
| WS | `/ws` | live score stream + browser mic audio in |
| GET | `/mic` | browser microphone capture page |

Two independent clients consume it: a Streamlit operator console (Python) and a
Chrome extension (JavaScript), sharing zero code.

**Hardware:** trained on 3 consumer GPUs — 1× RTX 3050, 2× RTX 4060.

---

## 3. Repository

`github.com/Anubhava-Jangwan/SONIX` · branch **`yugal/ml-pipeline`**
Local: `C:\Users\yugal\OneDrive\Desktop\Sonix`

```
realtime/            THE LIVE SYSTEM
  server.py            aiohttp: REST + WebSocket + /mic page
  engine.py            scoring loop, batches windows across calls
  session.py           per-call state machine, consent gate, VAD, audit
  models.py            model registry — which heads exist and where
  frontend.py          frozen wav2vec2, loaded once, cache-first
  ringbuffer.py        4.0 s windows at 0.5 s hop
  vad.py               silence gate
  live_ui.py           Streamlit operator console
  miccapture.py        browser mic capture panel
  source.py            WavFileSource (pull) / MicSource (push)
  selftest.py          END-TO-END LATENCY TEST — see §8
demo/                offline demo UI, 3 model tabs, replay mode
src/
  extract_embeddings.py  audio → 1024-dim vectors (the expensive step)
  train.py               train a head; --extra-emb-root STACKS (action="append")
  eval.py, metrics.py    EER, thresholds, score arrays
  calibration.py         temperature scaling
extension/           Chrome extension (second API client)
sonix_real/          LABELLED TEST SET — real vs cloned Indian public figures
outputs/models/      trained heads (~1 MB each, committed to git deliberately)
docs/                see §11
diagnose_pairs.py    THE DIAGNOSTIC — see §7
```

Root tools: `bench_clips.py`, `verify_head.py`, `inspect_ckpt.py`, `flag_rate.py`,
`augment_folder.py`, `make_indic_spoof.py`, `prep_indicsynth.py`, `prep_asvspoof5.py`,
`split_indic_holdout.py`, `stamp_labels.py`, `sonix_sdk.py`.

---

## 4. Data

### 4.1 Training corpus — exactly what the current head saw

| Embedding root | Bonafide | Spoof |
|---|---|---|
| `embeddings` — ASVspoof 2019 LA, clean | 2,580 | 22,800 |
| `embeddings_g711` — G.711 µ-law codec copies | 2,580 | 22,800 |
| `embeddings_rawboost` — RawBoost channel/impulsive | 2,580 | 22,800 |
| `embeddings_rirmusan_bonafide` / `_spoof` | 2,580 | 22,800 |
| `embeddings_indicvoices_tr` — genuine Indian speech | 37,032 | 0 |
| `embeddings_mms_tts` + `_aug` — Indic TTS spoof | 0 | 1,600 |
| `embeddings_indicsynth` + `_aug` — Indic VC spoof | 0 | 33,600 |
| **TOTAL** | **47,352** | **126,400** |

**Indic slice: 37,032 genuine : 35,200 synthetic = 1.05 : 1.**
Before this work it was 37,032 : **zero**. That ratio *was* the bug.

### 4.2 The Indic spoof corpus we built — 35,200 clips, 3 families, 12 languages

| Family | Rows | Languages |
|---|---|---|
| MMS-TTS (text-to-speech) | 1,600 = 800 clean + 800 channel-augmented | Hindi, Marathi, Bengali, Tamil, Telugu |
| IndicSynth (voice conversion) | 33,600 = 16,800 + 16,800 augmented | 12 Indian languages |

Full language list: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada,
Malayalam, Odia, Punjabi, Sanskrit, Urdu.

Channel augmentation on half of each family: G.711 µ-law round-trip, synthetic RIR
convolution reverb, pink noise at 8–30 dB SNR. Defaults in `augment_folder.py`:
p-reverb 0.8, p-noise 0.8, p-codec 0.5.

**This corpus did not exist publicly.** IndicVoices is bonafide-only; nobody had
built the balanced counterpart. It is a genuine contribution and should be presented
as one.

### 4.3 Held out — NEVER train on these

- **ASVspoof 2019 LA eval** — 71,237 clips (7,355 bonafide / 63,882 spoof)
- **ASVspoof 2019 LA dev** — 24,844 (2,548 / 22,296)
- **ASVspoof 2021 DF** — 75 % CM-key coverage; that caveat must appear beside any DF21 number
- **In-the-Wild** — real internet deepfakes, unseen tools. NOT YET EXTRACTED.
- **IndicVoices holdout** — 12 % of recordings, split by *recording id* not row id
- **`sonix_real/`** — see below

### 4.4 `sonix_real/` — the set that decides everything

`sonix_real/sonix_real/real/` and `/fake/` plus `clips_manifest.csv`.

- **9 named genuine** recordings: Modi, Rahul Gandhi, SRK, Bhuvan Bam,
  Ashish Chanchlani, Manmohan Singh, Nizamuddin, Rajat Sood, TVK Vijay
- **10 named clones** of the same speakers (+ Sonam) — made with **free consumer
  voice-cloning tools**
- **`real_01..10`** — team/friend phone-microphone recordings, English
- **`fake_01..10`** — free voice-clone downloads, English

**A matched real/clone pair from the same speaker is both the decisive test and the
most persuasive thing in any demo.**

---

## 5. The numbers ledger

### 🔒 LOCKED — safe to present

| Fact | Value |
|---|---|
| Architecture | frozen wav2vec2 XLS-R 300M + **262,657-param** MLP head |
| Windowing | 4.0 s window, 0.5 s hop, 87.5 % overlap |
| Band logic | 5-window moving average, hysteresis 3-of-5 |
| Thresholds | Amber 0.10, Red 0.90, operator-configurable |
| API surface | 7 REST endpoints + WebSocket, 2 independent clients |
| Padding bug — 1 s | 0.8795 → **0.0329** |
| Padding bug — 2 s | 0.5061 → **0.0005** |
| Padding bug — 3 s | 0.2224 → **0.0096** |
| Padding bug — full 66 s (control) | 0.1171 → 0.1171, unchanged |
| SNR shortcut — genuine that PASSED | 34.6, 37.7 dB → median 0.011, 0.002 |
| SNR shortcut — genuine that FAILED | 39.2–66.8 dB → median 0.779–1.000 |
| SNR shortcut — synthetic | 96–140 dB |
| Indic spoof corpus built | 35,200 embeddings, 3 families, 12 languages |
| Training corpus total | 47,352 bonafide / 126,400 spoof |
| Calibration temperature | T ≈ 1.05 |
| Calibration effect on EER | **none** — identical before and after (it was never meant to change EER) |

### ⚠️ CONTESTED — must be re-established before use

ASVspoof-2019 LA **eval** EER **1.4937 %** @ threshold 0.095130 for baseline
`head.pt`, eval set 71,237. Independently reproduced by two implementations to four
decimals on 1 Sept. **But `head.pt` was later overwritten during a merge** (outputs/
was gitignored, so git overwrote it without a conflict). Unverified until
`verify_head.py` passes. Also: dev EER 0.1592 % @ 0.836843 on the 24,844-clip dev set.

### ⏳ PENDING — placeholder only, with an owner

| Token | Owner | Blocked on |
|---|---|---|
| `[[EER_INDOMAIN]]` | Yugal | `verify_head.py` |
| `[[EER_DF21]]` | Navya | final model — must carry "(75 % key coverage)" |
| `[[FA_INDIC]]` | Navya | final model; per 4-second chunk, NOT per call |
| `[[INDIC_CLONE_DETECT]]` | Yugal | the `sonix_real/` gate test — unblocked, just not run |
| latency / real-time factor | Yugal | `realtime/selftest.py` — one command, never run |
| cost per call-minute | unassigned | 30 min of arithmetic once latency exists |

### ❌ NEVER SAY

"99 % accurate" · "detects any AI voice" · any latency figure (until measured) ·
"we have a production API" (say **"a working service interface"**) ·
**"We fixed Indian speech — 57 % → 0.02 %"** (see §6.3) · any number not 🔒 above.

---

## 6. THE OPEN PROBLEM — read this section carefully

### 6.1 Why 0.24 % dev EER does not mean 0.24 % of calls fail

`head_v3` scored **dev EER 0.24 % at epoch 31**. This is being misread as
"0.5 % of real audio fails". It is not that, and the gap is the whole problem.

- **Dev is a slice of the same corpus as training.** Same generators, same recording
  conditions, same languages, same channel augmentations. It shares attack types with
  the training set.
- A near-zero dev EER is therefore **expected**, not evidence of quality. It is a
  wiring check: it confirms labels aren't flipped and there's no leak.
- **Sanity rules for any EER:** single digits = working · ~40 % = label bug ·
  **~0 % = suspiciously easy test set or a leak.**
- The number that predicts real audio is **cross-dataset EER**, and in this field it
  is typically **10–30 %** for everyone, including published work. **We have never
  measured ours.**

So there is no contradiction between "0.24 %" and "it fails on my clips". They were
never measuring the same thing.

### 6.2 The observed failure

On a matched real/clone pair of the same Indian speaker:

| Model | Real Indian voice | Cloned Indian voice | Apparent behaviour |
|---|---|---|---|
| `head_robust_v2` | called FAKE ✗ | called FAKE ✓ | says "fake" to all Indian audio |
| `head_v3` | called REAL ✓ | called REAL ✗ | says "real" to all Indian audio |

Each got exactly one right — **and each got right the half that matched its constant
output.** That is the signature of a model that is not discriminating at all, and it
means combining / averaging / routing the two cannot help: two broken clocks showing
different times don't average into the right time.

**But two clips cannot prove that.** It could also be a model with real
discrimination whose threshold sits in the wrong place for this domain. Those look
identical on a green/red badge and are completely different problems.

### 6.3 The three candidate explanations

**(A) Language shortcut survived.** wav2vec2 XLS-R is multilingual; language identity
is strongly, linearly encoded in the embedding. IndicVoices is **bonafide-only** —
verified: `grep -c spoof scripts/extract_indicvoices.py` returns 0, one label
constant `BONAFIDE_LABEL = 0`. So "is this Hindi/Tamil/Bengali?" was a near-perfect
predictor of the label, and detecting a language is far cheaper than detecting
vocoder artifacts. If v3's 35,200 Indic spoofs didn't break that correlation, this is
what's happening.

**(B) Generator-generalisation gap — the most likely one.** `v3` learned Indian
synthesis from **MMS-TTS** (text-to-speech) and **IndicSynth** (voice conversion).
The `sonix_real/` clones are **zero-shot neural cloning from free consumer tools** —
a *third family* v3 has never seen. TTS artifacts ≠ VC artifacts ≠ zero-shot cloning
artifacts. Supporting evidence: v3's dev set *includes* Indic spoofs, and it scored
0.24 % there. A model that had purely learned "Indian = real" could not do that. So
it discriminates on *something* — the question is whether that something transfers.

**(C) Threshold misplacement — the cheapest outcome, and genuinely possible.**
`head_full_ho` flags genuine Indian speech at 0.02 %, i.e. it puts Indian audio very
low. Suppose Indian reals land at 0.001 and Indian clones at 0.08 — well separated,
high AUC, and **both below the 0.10 amber threshold.** The model would be working and
the threshold would be hiding it. You would never see this by looking at the band.

### 6.4 The one measurement that decides it

Stop looking at binary verdicts. Look at **score distributions** and compute **AUC**
per model on the Indian subset.

| AUC on Indian subset | Diagnosis | What to do |
|---|---|---|
| ≈ 0.50 | no discrimination — model emits a near-constant | routing/ensembling are dead; fix the training data |
| 0.65–0.90 | **the model CAN rank real vs fake — the operating point is wrong** | recalibrate; cheap wins available today |
| > 0.90 | it works and you've been misreading it | ship it |

**Also split by generator source:** if AUC is high on held-out MMS-TTS/IndicSynth and
≈ 0.5 on the free-tool clones, that is case (B) — a quantified statement of exactly
where your model's competence ends, which is a *good* result, not a bug.

---

## 7. THE FIX PATH — what to actually do

### Step 1 — run the diagnostic (already written, sitting in the repo)

`diagnose_pairs.py` is in the repo root. It embeds each clip once and scores all 7
heads on the cached vectors, then prints per-model, per-group AUC, EER and median
real/fake scores.

```bat
cd /d C:\Users\yugal\OneDrive\Desktop\Sonix
.venv\Scripts\python.exe diagnose_pairs.py --dir sonix_real\sonix_real
```

Then the control run, adding spoofs from your own generator families:

```bat
.venv\Scripts\python.exe diagnose_pairs.py --dir sonix_real\sonix_real --extra-fake data\indic_spoof
```

(Those are training data so the AUC there is optimistic — that's fine, you're only
checking whether v3 detects its own families *at all*.)

### Step 2 — fix the case you're actually in

**Case A — language shortcut survived (AUC ≈ 0.5 everywhere Indian).**
- **Group-balanced / worst-group loss.** Weight the four cells
  {Indian, non-Indian} × {real, fake} so the model cannot win by ignoring the
  Indian-fake cell. Sample weights on the existing loop — about an hour.
- **Domain-adversarial training (DANN / gradient reversal).** Add a second head that
  predicts *language* from the shared representation and reverse its gradient, so the
  representation is penalised for retaining language information. This is the textbook
  fix for this exact failure. ~1 day; you already have per-corpus language labels.
- **Diagnostic worth running regardless:** train a linear probe on the existing
  embeddings to predict *language*. High accuracy = language is linearly decodable =
  a 262k head can trivially use it as a shortcut. Minutes, and it turns "we think it
  learned language" into a measured claim.

**Case B — generator gap (AUC high on seen families, ≈ 0.5 on free-tool clones).**
- **Add a fourth generator family to training: free-tool clones.** You already made
  `fake_01..10` this way. Do it at scale, in Indian languages, across 2–3 different
  free tools, using **speakers that are not in `sonix_real/`** so that set stays a
  valid held-out test.
- Three families in training + a fourth unseen at test is also a far stronger
  experimental design, and it lets you report *measured degradation on an unseen
  generator* — which is what the field actually cares about.

**Case C — threshold misplacement (AUC high, verdicts wrong).**
- Compare `med real` vs `med fake` against the 0.10 / 0.90 thresholds. If both sit
  under 0.10 but are well separated, fit a **per-domain threshold**. Cheapest possible
  fix; honest as long as you say the thresholds are domain-conditional.

### Step 3 — getting to ONE model (the stated goal)

**Do not ship a router or an ensemble of the current heads.** Both robust_v2 and v3
appear to emit near-constants on Indian audio; a router would formalise the very
shortcut you are trying to remove (it would route on *language*, the spurious feature).

The path to one model is:

1. Run the diagnostic → identify the case
2. Apply the matching fix → train **one** new head (`head_v4`)
3. Validate on `sonix_real/` matched pairs + IndicVoices holdout + DF21
4. Register `v4` as `DEFAULT_KEY` in `realtime/models.py`
5. Keep the other 6 heads loaded **for comparison only** — that's the architecture
   story ("swap the judge in minutes"), not a production ensemble

**Optional, cheap, and worth trying once you have the diagnostic:** train a logistic
regression on the 7-dim vector of head scores (stacking). Minutes to fit. If it can't
beat the best single head, that's your evidence the heads carry no complementary
signal — which is itself a useful, reportable finding.

---

## 8. Running the system

`activate.bat` does **not** work on this machine — call the venv python directly.

```bat
REM Terminal 1 — the server
cd /d C:\Users\yugal\OneDrive\Desktop\Sonix
set HF_HUB_OFFLINE=1
.venv\Scripts\python.exe -m realtime.server --mode upload --auto-approve

REM Terminal 2 — the dashboard
cd /d C:\Users\yugal\OneDrive\Desktop\Sonix
.venv\Scripts\python.exe -m streamlit run realtime\live_ui.py

REM Offline demo UI (3 model tabs + replay mode)
.venv\Scripts\python.exe -m streamlit run demo\app.py
```

Wait for `Engine: front-end loaded from local cache` before uploading.
`HF_HUB_OFFLINE=1` stops transformers hunting HuggingFace. If port 8000 is held:
`taskkill /F /IM python.exe`.

Flags: `--ws-port 8001` · `--vad-energy 0.003` (quiet mic) · `--max-batch-size 4`
(small GPU) · `--mock` (no checkpoint).

**Latency — one command, never run, and you need this number:**
```bat
.venv\Scripts\python.exe -m realtime.selftest --ckpt outputs\models\head_v3.pt
```
Prints ms/batch, ms/window and PASS/FAIL against the 0.5 s hop budget. A recorded
data point exists in `server.py` help text: on a **GTX 1650**, batch of 8 overruns at
744 ms, batch of 4 fits at 394 ms. Never measured on the 4060.

**Training a head:**
```bat
.venv\Scripts\python.exe src\train.py ^
  --emb-root outputs\embeddings ^
  --extra-emb-root outputs\embeddings_g711 ^
  --extra-emb-root outputs\embeddings_rawboost ^
  --extra-emb-root outputs\embeddings_indicvoices_tr ^
  --extra-emb-root outputs\embeddings_indicsynth ^
  --out outputs\models\head_v4.pt
```
`--extra-emb-root` uses `action="append"`, so the roots genuinely stack.

---

## 9. The models

Seven heads registered in `realtime/models.py`, all present on disk, all sharing the
one frozen front-end.

| Key | File | Trained on | What is verified |
|---|---|---|---|
| `baseline` | `head.pt` | ASVspoof 2019 LA clean | 1.4937 % eval EER — **needs re-verification after a merge overwrite** |
| `augmented` | `head_aug.pt` | + G.711 µ-law | dev 0.24 % @ epoch 11 — **not comparable to baseline**, different test set |
| `robust` | `head_robust.pt` | + RawBoost | — |
| `robust_v2` | `head_robust_v2.pt` | + RIR / MUSAN | appears to have learned "Indian audio = fake" |
| `full_ho` | `head_full_ho.pt` | + IndicVoices, holdout-safe | appears to have learned "Indian audio = real". DF21 5.27 % vs baseline 9.48 %. Indian false-alarm 0.02 % — **this number measures nothing about detection** |
| `full_indic_as5` | `head_full_indic_as5.pt` | + IndicVoices + ASVspoof 5 (Navya's) | **nothing verified on our machine** |
| `v3` ← **DEFAULT** | `head_v3.pt` | everything + 35,200 Indic spoofs | dev EER 0.24 % @ epoch 31 — sanity check only |

Also on disk: `head_rebuilt.pt`, `head_aug_rebuilt.pt`, `temperature.npy`.

**Why v3 is default:** it is the only head with Indian languages on *both* sides of
the label. That is a reasoning-based choice, **not** a verified head-to-head.

---

## 10. Engineering history — bugs found and fixed

Two of these silently invalidated results, so the dates matter.

1. **Missing `import torch` in `engine.py`.** `_head_forward` used torch but the
   module never imported it. The `NameError` fired inside `asyncio.to_thread`, escaped
   `run()`'s CancelledError-only handler, and **killed the scoring task on the first
   batch**. The server kept accepting audio and scored nothing, silently, until
   restart. Fixed, plus per-batch isolation and a done-callback that logs loudly.

2. **`_trim_pending()` subtracted an int from `None`.** `max_pending_windows` is
   deliberately `None` for uploads ("score every window"). The `TypeError` fired inside
   `push_audio` on the **first window that passed the gate**, before `window_count` was
   incremented, and killed the feed task. Result: a 62-window clip returned **one**
   score, filed under index **−1**, and the dashboard drew a confident GREEN band from
   it. **Every upload verdict collected before 3 September is void.**

3. **`live_ui` reported stalled runs as complete.** The stall warning was written into
   the same `st.empty()` placeholder the success line later overwrote. Fixed; partial
   runs now render as INCOMPLETE with ring-buffer counts, gate counts and the engine's
   last error.

4. **Front-end load appeared to hang.** transformers 404s on `main/model.safetensors`
   and then walks the repo's open pull requests looking for one, silently pulling
   ~1.2 GB. Fixed with `local_files_only=True` first, falling back to download.

5. **`head.pt` silently overwritten in a merge** — `outputs/` was gitignored, and git
   overwrites ignored files without conflict. This is why the 1.4937 % headline is
   contested. `.gitignore` now lets `outputs/models/*.pt` through deliberately.

6. **Other environment facts:** the Windows checkout is CRLF while the repo stores LF
   (`core.autocrlf` comes from Windows global config) — committing from a Linux shell
   without `-c core.autocrlf=true` rewrites the whole tree. OneDrive syncing `.git/`
   causes recurring stale `index.lock`.

---

## 11. Docs and deck assets already produced

In `docs/`:

| File | What |
|---|---|
| `SONIX_MASTER.md` | project master reference |
| `SONIX_National_Submission_Pack.pdf` | 28 pages — project A–Z, six-slide build spec, 11 figures, researched finale intelligence, Q&A bank |
| `SONIX_deck_assets.zip` | 11 diagrams as editable SVG + 2× PNG |
| `PRESENTATION_SCRIPTS.md` / `SONIX_Presentation_Scripts.docx` | per-speaker timed scripts, 4 min slides + 6 min demo |
| `RESULTS_TABLE.md` | Yukti's reproducible-numbers source |
| `API.md` | integration reference |
| `slides/sonix_technical_approach.svg` / `.png` | the technical-approach one-pager |

### SIH submission facts — verified from official sources

- **The deck is SIX slides**, including the title, exported to **PDF**. No PPT/DOCX.
  Template must be used unmodified.
- The six fixed sections: Title · Proposed Solution · Technical Approach ·
  Feasibility & Viability · Impact & Benefits · Research & References.
- **SIH publishes NO percentage weights.** A rubric of 25/20/20/20/15 circulates on
  blogs — **it is fabricated**, from a site that also wrongly claims 10 slides.
- Official criteria, verbatim from sih.gov.in: *"novelty of the idea, complexity,
  clarity and details in the prescribed format, feasibility, practicability,
  sustainability, the scale of impact, user experience and potential for future work
  progression."*
- **Judging is multi-round with the same jury**, scoring the delta between visits.
  Hold work back to land between rounds rather than arriving finished.
- *"Potential for future work progression"* is an official criterion — the roadmap is
  a scored field and the honest home for the ambitious version.
- A 2024 winning deck on a security PS wrote **"80 % completed with testing/validation
  pending"** on its own technical slide.
- Team is currently **on the waitlist** for the national submission.

---

## 12. Team

| Person | Owns |
|---|---|
| **Yugal** (Croakie) | ML pipeline, training, live engine, integration, branch `yugal/ml-pipeline` |
| **Navya** | RIR/MUSAN augmentation, IndicVoices, ASVspoof 5, DF21 evaluation |
| **Akshat** | IndicSynth corpus, In-the-Wild, labelled recordings |
| **Anubhav** | Data extraction, mic-capture panel, `sonix_real/` clip set |
| **Suryansh** | Demo UI, replay mode, Chrome extension |
| **Yukti** | Metrics, `RESULTS_TABLE.md`, presentation |

---

## 13. What to do next, in order

1. **Run `diagnose_pairs.py`** on `sonix_real/` (+ the `--extra-fake` control).
   Everything branches on this. ~2 minutes of compute.
2. **Run `verify_head.py`** — re-establish or retire the 1.4937 % headline. ~15 min.
3. **Run `realtime/selftest.py`** — get the real-time factor. ~10 min. This is the
   number the PS title demands and it does not exist yet.
4. Apply the §7 fix matching the diagnosed case → train **one** `head_v4`.
5. Validate v4 on `sonix_real/`, IndicVoices holdout, DF21.
6. Cost/concurrency arithmetic (30 min, once #3 exists).
7. Ablations: no-Indic-spoof / MMS-TTS only / IndicSynth only / full.
8. Record a backup demo video; freeze code two hours before any demo.

---

## 14. How I want to work (carry this into the new chat)

- Casual and direct. Concise over elaborate. Honest over accommodating.
- For anything I'm building myself: explain the pipeline and the steps first, then
  code in blocks with explanation — not a finished file dumped in one go.
- Never fabricate a result to make a slide look better. If something is unverified,
  say so and mark it `[[PLACEHOLDER]]`.
