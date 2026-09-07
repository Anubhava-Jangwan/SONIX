# SONIX — everything needed to build the PPT

**SIH 2026 · Problem Statement SIH26104 · Team SONIX**
**Supersedes YUKTI_PPT_REFERENCE.md, PPT_STATUS.md and DEPLOYMENT_MODEL.md — this is the one file.**

Last updated **3 Sept 2026, early morning**. `git pull` and re-read §8 before every deck revision.

---

## 1. How to use this

Paste this whole file into Claude as context, then ask for the deck. Four rules it must obey:

1. **Never invent a number.** §8 lists what is safe. Everything else is a `[[PLACEHOLDER]]`.
   A blank is fine; a wrong number in front of a jury is fatal.
2. **Diagrams beat text.** Evaluators scan visually first. §10 specifies the exact visual per slide.
3. **The deck is judged twice, by two different audiences.** §2 — this governs everything.
4. **SONIX is a security layer with an API, not an app.** §5 — most teams get this wrong.

---

## 2. Two audiences, and why one wins

**Audience A — the portal reviewer.** 114 teams narrow to 45; those 45 upload this deck and SIH
picks the top 5 **from the PPT alone**. Nobody in the room. No narration. No demo.

**Audience B — the live jury.** 4 min slides, 6 min demo, 5 min questions.

**Optimise for A.** It is the round that eliminates you, and a self-contained slide still works
when someone talks over it — the reverse is not true.

**Test for every slide: is it understandable with the sound off?** Every chart gets a caption
stating its *conclusion*, not its axes. Every diagram is labelled for a reader who has never met us.

---

## 3. What evaluators score

| Weight | Criterion | What it means for us |
|---|---|---|
| **25 %** | Innovation & uniqueness | Highest weight → §6 |
| 20 % | Problem understanding | Show we understand *Indian* voice fraud specifically |
| 20 % | Technical feasibility | Named components, real numbers, buildable architecture |
| 20 % | Impact & scalability | Who benefits, at what scale, at what cost |
| 15 % | Presentation quality | Necessary, smallest slice — don't spend all night on gradients |

**Rejection patterns:** vagueness about *how*, drifting off the problem statement, AI buzzwords
with no substance.

**Format rules:** diagrams over text · **max 6 bullets/slide** · **min 14 pt font**.

> ⚠️ **Yugal must confirm the official SIH template from the portal.** Slide counts have varied
> between years. This document gives content and visuals; they map onto whatever skeleton is official.

---

## 4. What SONIX is

### The problem
AI voice cloning is cheap and convincing from seconds of audio. Attackers impersonate CXOs, bank
officials and family members to authorise transfers, extract OTPs, or bypass verification. India
is a large target: UPI fraud, family-emergency scams, official impersonation. **Telephony has no
way to tell a cloned voice from a real one during the call.**

### The solution
SONIX scores a live call for the probability the voice is synthetic and surfaces it as
Green / Amber / Red. **It never blocks a call. It flags for a human.**

### The pipeline — be exact, vagueness costs 20 %
```
live audio (16 kHz mono)
  → 4.0 s windows at 0.5 s hop            (87.5 % overlap)
  → silence gate                           (near-silent windows dropped)
  → wav2vec2 XLS-R 300M, FROZEN ❄         → mean-pool → 1024-dim
  → MLP head, TRAINED 🔥                   1024 → 256 → ReLU → Dropout(0.3) → 1
                                           262,657 parameters
  → sigmoid                                → P(synthetic) 0–1
  → 5-window moving average
  → hysteresis: 3 of 5 must agree to switch band
  → 🟢 GREEN / 🟡 AMBER / 🔴 RED
```

---

## 5. Deployment: an API and SDK, not an application

**This is the section most teams will miss, and the PS states it three times.**

> *"expose **APIs and SDKs** for seamless integration with banking applications, enterprise
> communication systems, and telecom operator infrastructures"*
> *"**REST/gRPC APIs and SDKs** for integration with core banking systems, contact center
> platforms, enterprise communication tools, and telecom networks"*
> *"A **reusable security layer** for telecom operators and enterprises"*

SONIX is **middleware**. Banks, contact centres and telcos call it from inside systems they
already run. The dashboard is a *client*, not the product.

| Layer | What it is | How to present it |
|---|---|---|
| Scoring engine | Frozen front-end + trained head, batched across concurrent calls | The core IP |
| **Integration API** | 7 REST endpoints + WebSocket, consent-gated | **The deliverable** |
| Operator console | Streamlit dashboard, live risk timeline | A reference client |
| Chrome extension | Browser-call capture | A second client — proves it's platform-agnostic |

### The slide that proves we read the PS: three lines of integration code

```python
from sonix_sdk import SonixClient
sonix = SonixClient("http://sonix.internal:8000")

result = sonix.score_file("call_88213.wav")
if result.band == "RED":
    hold_transaction(reason=f"voice-clone risk {result.mean:.0%}")
```

**Put this on slide 3.** A judge who reads it believes the integration story instantly.

### The strongest honest claim
**Two independent clients already consume the same endpoints** — a Python dashboard and a
JavaScript Chrome extension, sharing no code. That is the real test of whether something is an
interface or just an app's backend, and most hackathon "APIs" fail it.

### Say this during the demo
> *"The dashboard you're watching is one client of our API. A bank would call the same endpoints
> from inside their own contact-centre software."*

### Maturity — state it yourself before a judge does
We have a working service interface: 7 REST endpoints, WebSocket streaming, proper status codes,
two independent clients. **What it is not yet:** authenticated, versioned, or a published SDK.
Those are days of work, not architecture changes — the separation that makes them cheap already
exists. Drawing that line yourself earns more credibility than claiming a finished API.

---

## 6. Innovation — the 25 % slice

Do not present "we trained a deepfake detector". Every team presents that. **Four real differentiators:**

**① Streaming, not batch.** Nearly every published detector classifies a whole file offline. We
run on a live stream with overlapping windows, a silence gate, temporal smoothing and hysteresis —
the machinery that makes it deployable rather than a notebook result.

**② Frozen front-end, tiny trainable head.** 300M frozen params listen; 262,657 trained params
decide. A new cloning tool appears → extract embeddings once, retrain in **minutes**. An
architecture chosen for a threat landscape that changes weekly. `/api/models` serves several
heads behind one endpoint with no extra GPU memory.

**③ Consent and audit designed in from day one.** Pairing-code consent gate, immutable per-call
audit trail, no audio retention. **Enforced in the state machine** — `push_audio()` drops frames
unless the call is approved. Most projects ignore this entirely.

**④ We audit our own model and fix root causes.** The rarest one — §7.

---

## 7. The story spine: we find our own failures and fix the cause

This is what makes a practitioner jury believe the rest. Anyone shows a good number. Almost nobody
shows a measured before-and-after of a bug they found themselves.

**Case 1 — the padding bug.** The model listens in 4-second chunks. Shorter clips were padded with
*digital silence*, so a 1-second clip was three-quarters nothing — and the model called real people
fake. We now repeat the audio instead.

| Clip length | Before (zero-pad) | After (repeat-pad) |
|---|---|---|
| 1 second | **0.8795** | **0.0329** |
| 2 seconds | 0.5061 | 0.0005 |
| 3 seconds | 0.2224 | 0.0096 |
| Full 66 s (control) | 0.1171 | 0.1171 — unchanged |

**Case 2 — the background shortcut.** Genuine recordings still flagged. Instead of tuning
thresholds we investigated: the model had learned **"clean background = fake"**, not synthesis
detection. ASVspoof's silence is digitally dead at −75 dBFS, a floor no real microphone produces.

| | SNR | Median score |
|---|---|---|
| Passing genuine clips | 34.6, 37.7 dB | 0.011, 0.002 |
| Failing genuine clips | 39.2 – 66.8 dB | 0.779 – 1.000 |
| Synthetic clips | 96 – 140 dB | — |

Phone noise-suppression strips room tone, so genuine voices looked synthetic. Fixed in the
**training data** — room reverb and real-world noise, so background can no longer be a cue.

**Case 3 — the Indian-speech blind spot.** The PS explicitly demands *"diverse Indian accents and
dialects"*, so we tested it. **The majority of genuine Indian speech was flagged as fake.** We
diagnosed why (no Indian data in training) and are fixing it — §8.

**Frame all three as:** observation → hypothesis → measurement → root-cause fix → verification.

---

## 8. LOCKED vs PENDING — the numbers ledger

### 🔒 LOCKED — safe on a slide

| Fact | Value |
|---|---|
| Padding bug, 1 s / 2 s / 3 s / 66 s | 0.8795→0.0329 · 0.5061→0.0005 · 0.2224→0.0096 · 0.1171 unchanged |
| SNR shortcut split | genuine pass 34.6–37.7 dB · genuine fail 39.2–66.8 dB · synthetic 96–140 dB |
| Architecture | frozen wav2vec2 XLS-R 300M + **262,657-param** MLP head |
| Windowing | 4.0 s window, 0.5 s hop, 87.5 % overlap |
| Band logic | 5-window moving average, hysteresis 3-of-5 |
| Thresholds | Amber 0.10, Red 0.90, operator-configurable |
| API surface | 7 REST endpoints + WebSocket; 2 independent clients |
| **Indic spoof corpus built** | **35,200 embeddings** across **5 synthesis conditions** |
| — MMS-TTS (5 languages) | 1,600 (800 clean + 800 channel-augmented) |
| — IndicSynth (12 languages) | 33,600 (16,800 clean + 16,800 channel-augmented) |
| Languages covered | Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi, Sanskrit, Urdu |

⚠️ **Always state: scores are 0–1 and HIGHER = "model thinks it's fake."**

### ⏳ PENDING — placeholders only

| Token | Owner | Blocked on |
|---|---|---|
| `[[MODEL_NAME]]` | Yugal | training data transfer |
| `[[EER_INDOMAIN]]` | Yugal | old 1.4937 % came from a checkpoint overwritten in a merge — **do not quote it**, being re-verified |
| `[[EER_DF21]]` | Navya | final model. Must carry **"(75 % key coverage)"** |
| `[[FA_REAL]]` | Yugal | final model + Akshat's clips |
| `[[FA_INDIC]]` | Navya | final model. Per 4-second chunk, **not** per call |

### ⛔ BLOCKED — plan the slide to survive without

| Token | Why | Plan B |
|---|---|---|
| `[[EER_ITW]]` | In-the-Wild not extracted | "⏳ measurement in progress" row |
| Indian-clone detection result | No cloned Hindi test clips yet | **"Our next measurement"** — never a claim |
| Cost / scale figures | Nobody assigned | Someone needs 30 min. Winners cite this as a differentiator |
| Latency | Never measured | Do not quote a number |

### ⚠️ The claim that must not appear

We measured **57 % of genuine Indian speech flagged as fake**, then trained a model that cut it to
**0.02 %**. Both real. **But that model also passes cloned Indian voices as genuine** — every
Indian sample it saw was real, so it learned "Indian audio = real" rather than learning to detect.

- ✅ *"We discovered our detector fails on Indian speech, measured it, diagnosed the cause, and
  built a 35,200-clip Indian synthetic-speech corpus across three generator families to fix it."*
- ❌ *"We fixed it — 57 % → 0.02 %."*

If verified before submission, this moves to 🔒. Until then it does not.

### Team data status

| Source | Owner | Status |
|---|---|---|
| ASVspoof 2019 LA + G.711 + RawBoost | Yugal | ✅ |
| MMS-TTS Indic spoof + augmented | Yugal | ✅ 1,600 rows, verified |
| IndicSynth Indic spoof + augmented | Akshat | ✅ 33,600 rows, verified |
| RIR/MUSAN augmented | Navya | ✅ built, transfer pending |
| **IndicVoices (genuine Indian speech)** | Navya | ⛔ **transfer pending — blocks training** |
| MLAAD | Navya | ⛔ gated, awaiting author approval |
| Hindi real + cloned recordings (TEST) | Akshat | ⏳ not started — **decides whether the fix worked** |

---

## 9. Presenting unfinished results without looking incomplete

**Do not leave blank cells.** Show the *evaluation framework* with status markers:

| Benchmark | What it tests | Status |
|---|---|---|
| ASVspoof 2019 LA eval | In-domain accuracy | `[[EER_INDOMAIN]]` |
| ASVspoof 2021 DF | Cross-dataset generalisation | `[[EER_DF21]]` (75 % coverage) |
| In-the-Wild | Real internet deepfakes, unseen tools | ⏳ in progress |
| Indian-language corpora | Language robustness | ⏳ in progress |
| Our labelled recordings | Real-world false alarms | `[[FA_REAL]]` |

Caption:
> *Evaluation is ongoing. Cross-dataset EER for audio anti-spoofing is typically 10–30 % —
> degradation is the known open problem in this field, and we measure it rather than avoid it.*

**Why this works.** It shows you know the right benchmarks and what an honest result looks like.
Stating the 10–30 % expectation first inoculates you — a reviewer cannot use it against you.
Against five decks claiming 99 %, the one naming its open problems stands out.

---

## 10. Visual specification — build these

**Slide 1 — Title.** Official SIH fields only. PS ID **SIH26104**, theme, category (Software),
Team ID, Team Name. No visual.

**Slide 2 — Idea / Solution.**
*Primary visual: the attack-and-defence flow*, horizontal, five steps:
`[Scammer] → [AI clone] → [Victim's phone] → [SONIX 🟢🟡🔴] → [Human decides]`
Real icons, not ASCII. **Colour the band Green/Amber/Red — it's our product identity, use it on
at least three slides.** Max 6 bullets: real-time not post-call · flags never blocks ·
consent-gated · thresholds configurable per organisation.

**Slide 3 — Technical Approach.** *Two visuals, side by side:*
- **The pipeline** — horizontal, each stage boxed with its data shape. **Snowflake ❄ on the frozen
  front-end, flame 🔥 on the trained head.** Makes the architecture argument in one glance.
- **The integration diagram** — our API in the middle, arrows out to core banking / contact centre /
  telecom / collaboration platforms. **This is the slide that proves we read the PS.**

Plus the three-line SDK snippet from §5. Stack in small type: Python, PyTorch, transformers,
Streamlit, aiohttp WebSockets, Chrome extension.

**Slide 4 — Feasibility & Viability.** *Our strongest slide, two charts:*
- **Chart A — the padding fix.** Line chart, X = clip length, Y = fake-score 0–1, two lines
  (before/after), dashed horizontal line at 0.10 labelled "Amber threshold". Y-axis in plain words:
  *"model's fake score (higher = thinks it's fake)"*. Caption: *"A 1-second clip of a real human
  went from confidently fake to confidently real."*
- **Chart B — the shortcut.** Scatter, X = SNR dB, Y = median score, genuine vs synthetic in two
  colours, vertical line at ~38 dB. Caption: *"The model had learned 'clean background = fake',
  not synthesis detection. We found it, proved it, fixed it in the training data."*

Then the benchmark matrix from §9 and a short risk/mitigation list.

**Slide 5 — Impact & Benefits.** *Primary visual: stakeholder map* — individuals, banks and call
centres, telecom operators, government helplines, with the benefit named for each. Give privacy
real space: consent gate, audit trail, no retention, feature-only logging. Avoid unattributed
statistics.

**Slide 6 — Research & References.** Two columns. **Datasets:** ASVspoof 2019 LA, ASVspoof 2021 DF,
In-the-Wild, IndicVoices, IndicSynth, MLAAD. **Papers:** Müller et al. INTERSPEECH 2022;
wav2vec2 / XLS-R; RawBoost; MLAAD. Note MLAAD is CC BY-NC.

**Design rules everywhere:** max 6 bullets · min 14 pt · one primary visual per slide ·
Green/Amber/Red reserved for the risk band only · every chart captioned with its conclusion.

---

## 11. The 6-minute demo — script it

Longer than the slide time. The organisers are saying they care more about it working.

1. **(30 s)** Dashboard at rest. Point at the model selector and threshold sliders. **Say the API
   sentence from §5.**
2. **(90 s)** Upload a genuine recording. Graph builds window by window. Stays GREEN.
3. **(90 s)** Upload a cloned recording **of the same speaker**. Climbs to RED. *The matched pair
   is the moment that lands.*
4. **(60 s)** Switch models in the sidebar, re-run the same clip. Proves the architecture claim.
5. **(60 s)** Consent gate and audit trail.
6. **(30 s)** Buffer.

**Record a backup video of a clean run.** If the laptop dies, the video *is* the demo. Winners'
retrospectives are unanimous that teams who skip this regret it. **Freeze code two hours before.**

---

## 12. Jury Q&A

**"Does it work on any audio?"** In-domain `[[EER_INDOMAIN]]`. Cross-dataset it degrades — the
acknowledged open problem, and exactly why SONIX never blocks a call.

**"Can it detect a cloned Indian-language voice?"** We proved we don't false-alarm on genuine
Indian speech and we catch Western synthetic speech. We have built a 35,200-clip Indian synthetic
corpus across three generator families; **detecting a cloned Hindi voice is our next measurement.**
Do not claim it works.

**"How would a bank deploy this?"** Self-hosted container inside their infrastructure — audio never
leaves their network. They call our REST endpoint or open a WebSocket from their contact-centre
platform. Thresholds configured per workflow: stricter for fund transfers than general enquiries.
The console is a reference client; in production the score feeds their fraud engine.

**"Why EER and not accuracy?"** Benchmark data is ~90 % spoof. "Fake" every time scores 90 %
accuracy and is useless. EER is the field standard.

**"What when a new cloning tool appears?"** Front-end frozen, 262k head trained. Extract embeddings
once, retrain in minutes, serve it from `/api/models` — no redeployment, no downtime.

**"Privacy?"** Consent gate enforced in the state machine, immutable audit trail, no audio retained.
Designed in, not bolted on.

**"Cost and scale?"** ⚠️ **Unprepared. Someone must do 30 minutes on GPU cost per call-minute,
concurrent calls per instance, deployment cost.** Winners cite financial literacy as the differentiator.

**If you don't know, say so and say what you'd measure.** To a practitioner jury that reads as
competence.

---

## 13. Never say

❌ "99 % accurate" · ❌ "We fixed Indian speech, 57 % → 0.02 %" · ❌ "Detects any AI voice" ·
❌ Any latency figure · ❌ "We have a production API" (say "a working service interface") ·
❌ Any number not 🔒 in §8

---

## 14. Sources

- SIH 2026 PPT template & evaluator scoring — https://blogs.reskilll.com/sih-2026-ppt-template-exact-format-slides-evaluators-score/
- SIH idea presentation format — https://www.lets-code.co.in/blogs/sih-2025-complete-guide-ppt-template/
- SIH 2024 winners' retrospective — https://how-we-won-sih-24-and-survived-it.hashnode.dev/everything-about-winning-sih-2024
- Müller et al., "Does Audio Deepfake Detection Generalize?" — https://arxiv.org/pdf/2203.16263
- MLAAD — https://deepfake-total.com/mlaad
- IndicSynth — https://huggingface.co/datasets/vdivyasharma/IndicSynth

---

## 15. Final checklist

- [ ] Official SIH template confirmed from the portal and followed exactly
- [ ] Every slide readable **with the sound off**
- [ ] Max 6 bullets, min 14 pt, one primary visual per slide
- [ ] Slide 3 has **both** the pipeline diagram and the integration diagram + SDK snippet
- [ ] Slide 4 has both charts (padding fix, SNR shortcut)
- [ ] "Higher score = model thinks it's fake" stated wherever scores appear
- [ ] Every DF21 figure carries "(75 % key coverage)"
- [ ] Unfinished results shown as a benchmark matrix with status, never blank cells
- [ ] Every `[[PLACEHOLDER]]` in a table at the end with an owner
- [ ] Green/Amber/Red only for the risk band
- [ ] Demo script rehearsed; backup video recorded
- [ ] Cost/scale answer prepared
- [ ] Indian-clone Q&A answer agreed by the whole team

---

## 16. Change log

- **3 Sept, early morning** — Indic spoof corpus complete: 35,200 embeddings, 12 languages, 3
  generator families (MMS-TTS, IndicSynth voice conversion, plus channel augmentation), all
  verified label 1. Deployment/API section added — SONIX is middleware, not an app. `sonix_sdk.py`
  and `docs/API.md` added to the repo.
- **2 Sept, evening** — Ledger created. MMS-TTS Indic spoofs complete.
