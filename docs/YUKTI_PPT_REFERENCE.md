# SONIX — PPT reference for Yukti

**SIH 2026 · Problem Statement SIH26104 · Team SONIX**
**Written 2 Sept 2026. Hand this entire file to your AI assistant before it writes anything.**

---

## 0. How to use this document

Paste this whole file into Claude as context, then ask for the deck. Three rules it must obey:

1. **Never invent a number.** §6 lists exactly which figures are safe. Everything else is a
   `[[PLACEHOLDER]]` filled in at the end. A blank is fine; a wrong number in front of a jury
   is fatal.
2. **Diagrams beat text.** Evaluators scan visually before they read. §7 specifies the exact
   visual for every slide — build those, don't write paragraphs.
3. **The deck is judged twice, by two different audiences.** §1. This governs everything.

---

## 1. The thing most teams get wrong: this deck has TWO audiences

**Audience A — the portal reviewer.** 114 teams narrow to 45; those 45 upload this deck and
SIH picks the top 5 **from the PPT alone**. Nobody is in the room. No narration. No demo. The
slide has to make its whole argument by itself.

**Audience B — the live jury.** 4 minutes of slides, 6 minutes of demo, 5 minutes of
questions. Here the deck supports a speaker.

**These pull in opposite directions.** A deck built for live presentation is sparse — the
speaker carries it. A deck built for cold reading is dense — it must stand alone. Optimise for
**Audience A**, because that is the round that eliminates you, and because a self-contained
slide still works fine when someone is talking over it. The reverse is not true.

**Practically:** every slide must be understandable with the sound off. Every chart gets a
one-line caption stating its conclusion, not just its axis labels. Every diagram is labelled
so a reader who has never met us can follow it.

---

## 2. What evaluators actually score

| Weight | Criterion | What that means for us |
|---|---|---|
| **25 %** | Innovation & uniqueness | Highest weight. What makes SONIX different from "we fine-tuned a classifier"? §4. |
| **20 %** | Problem understanding & clarity | Show we understand *Indian* voice-fraud specifically, not deepfakes in general. |
| **20 %** | Technical feasibility | Named components, real dimensions, an architecture that could actually be built. |
| **20 %** | Impact & scalability | Who benefits, at what scale, at what cost. |
| **15 %** | Presentation quality | Clean, visual, readable. Necessary but the smallest slice — do not spend all night on gradients. |

**Decks get rejected for:** vagueness about *how* it works, drifting off the problem
statement, and AI buzzwords with no technical substance behind them. All three are avoidable.

**Formatting rules from the evaluation guidance:** diagrams over text, **max 6 bullets per
slide**, **minimum 14 pt font**.

> ⚠️ **Use the official SIH template file for our round.** The prescribed slide count has
> varied between years (6 for idea submission, more in some formats). Yugal must confirm the
> exact template from the portal before the deck is finalised — this document describes
> content and visuals, which map onto whatever the official skeleton is.

---

## 3. What SONIX actually is

### The problem
AI voice cloning is cheap, fast and convincing. A scammer clones a family member's or a bank
official's voice and calls a victim — "send money now", "verify this OTP". India is a large
target: UPI-based fraud, family-emergency scams, impersonation of officials. Telecom
infrastructure today has no way to tell a cloned voice from a real one **during** the call.

### The solution
SONIX scores a live call for the probability the voice is synthetic and surfaces it to a human
as a traffic light — Green / Amber / Red. **It never blocks a call. It flags for a person.**

### The pipeline (be exact — vagueness here costs 20 %)
```
live audio (16 kHz mono)
  → 4.0 s windows at 0.5 s hop           (87.5 % overlap)
  → silence gate                          (near-silent windows dropped before scoring)
  → wav2vec2 XLS-R 300M, FROZEN           → mean-pool → 1024-dim embedding
  → MLP head, TRAINED                     1024 → 256 → ReLU → Dropout(0.3) → 1
                                          262,657 parameters
  → sigmoid                               → P(synthetic), 0–1
  → 5-window moving average
  → hysteresis: 3 of 5 must agree to switch band
  → GREEN / AMBER / RED
```

### What is built and working
- **Live streaming dashboard** — upload a call or capture live mic; the risk score builds on a
  graph window by window, in real time.
- **Five detector models switchable at runtime**, all sharing one front-end.
- **Consent gate + immutable audit trail** — pairing code before any scoring.
- **Operator-configurable thresholds**, live in the UI.
- **Chrome extension** for browser-based calls.

---

## 4. Innovation — the 25 % slice, and where we actually win

Do not present this as "we trained a deepfake detector." Every team presents that.

**Four things genuinely differentiate SONIX. Lead with them.**

**① Streaming, not batch.** Almost every published detector classifies a whole audio file
offline. SONIX runs on a live stream with overlapping windows, a silence gate, temporal
smoothing and hysteresis — the machinery that makes it a *deployable system* rather than a
notebook result. Say this explicitly; it is the clearest gap between us and the field.

**② Frozen front-end, tiny trainable head.** 300M frozen parameters do the listening; 262,657
trained parameters do the deciding. Consequence: a new cloning tool appears, we extract
embeddings once and retrain in **minutes**, not days. That is an architecture chosen for a
threat landscape that changes weekly.

**③ Designed for consent and audit from day one.** Voice monitoring is legally sensitive. A
pairing-code consent gate and an immutable per-call audit trail were in the first design, not
retrofitted. Most hackathon projects ignore this entirely.

**④ We audit our own model and fix root causes.** This is the strongest and rarest one — §5.

---

## 5. The story spine: we find our own failures and fix the cause

This is what makes a practitioner jury believe the rest of the deck. Anyone can show a good
number. Almost nobody shows a measured before-and-after of a bug they found themselves.

**Two documented cases. Both are locked and safe to present.**

**Case 1 — the padding bug.** The model listens in 4-second chunks. Clips shorter than that
were being padded with *digital silence*, so a 1-second clip became three-quarters nothing —
and the model called real people fake. We now repeat the audio instead.

| Clip length | Before (zero-pad) | After (repeat-pad) |
|---|---|---|
| 1 second | **0.8795** | **0.0329** |
| 2 seconds | 0.5061 | 0.0005 |
| 3 seconds | 0.2224 | 0.0096 |
| Full 66 s (control) | 0.1171 | 0.1171 — unchanged |

**Case 2 — the shortcut.** Genuine recordings were still being flagged. Instead of tuning
thresholds we investigated, and found the model had learned **"clean background = fake"** —
not synthesis detection at all. ASVspoof's silence is digitally dead at −75 dBFS, a floor no
real microphone produces. Our real clips split almost perfectly at ~38 dB SNR:

| | SNR | Median score |
|---|---|---|
| Passing genuine clips | 34.6, 37.7 dB | 0.011, 0.002 |
| Failing genuine clips | 39.2 – 66.8 dB | 0.779 – 1.000 |
| Synthetic clips | 96 – 140 dB | — |

Phone noise-suppression strips room tone, so genuine voices ended up looking synthetic. We
fixed it in the **training data** — room reverb and real-world noise so the model can no
longer use background as a cue.

**Frame it as:** *observation → hypothesis → measurement → root-cause fix → verification.*
That is engineering method, and it is the deck's centre of gravity.

---

## 6. LOCKED vs NOT LOCKED — read before writing any number

### ✅ LOCKED — safe on a slide

| Fact | Value |
|---|---|
| Padding bug, 1 s clip | 0.8795 → 0.0329 |
| Padding bug, 2 s / 3 s / 66 s | 0.5061 → 0.0005 · 0.2224 → 0.0096 · 0.1171 unchanged |
| SNR shortcut split | genuine pass 34.6–37.7 dB; genuine fail 39.2–66.8 dB; synthetic 96–140 dB |
| Architecture | frozen wav2vec2 XLS-R 300M + 262,657-param MLP head |
| Windowing | 4.0 s window, 0.5 s hop, 87.5 % overlap |
| Band logic | 5-window moving average, hysteresis 3-of-5 |
| Thresholds | Amber 0.10, Red 0.90, operator-configurable |

**Always state: scores run 0–1 and HIGHER MEANS "model thinks it is fake."** Without that line
a reader sees 0.88 and assumes 88 % confident it's genuine.

### ⏳ NOT LOCKED — placeholders

| Figure | Placeholder |
|---|---|
| In-domain EER (ASVspoof 2019 LA) | `[[EER_INDOMAIN]]` |
| Cross-dataset EER (ASVspoof 2021 DF) | `[[EER_DF21]]` — must always carry "(75 % key coverage)" |
| In-the-Wild EER | `[[EER_ITW]]` |
| Real-clip false-alarm rate | `[[FA_REAL]]` |
| Indian-speech false-alarm rate | `[[FA_INDIC]]` |
| Final model name | `[[MODEL_NAME]]` |

### ⚠️ The Indic finding — the one to get right

We measured that our detector flagged **the majority of genuine Indian speech as fake**. Real,
measured, and the most important thing we learned.

We then trained a model that cut that to near zero — **but that model also passes cloned
Indian voices as genuine**, because all the Indian speech in training was real, with no Indian
fakes to learn from. It learned "Indian audio = real" rather than learning to detect.

- ✅ Present: *"we discovered our system fails on Indian speech, measured it, diagnosed why,
  and are fixing it with multi-generator Indian synthetic data."*
- ❌ Do not present: *"we fixed it, 57 % → 0.02 %."* That number describes a model that cannot
  catch an Indian deepfake.
- If a judge asks whether we catch a cloned Hindi voice: **"that is our next measurement, not
  a claim we make."** Agree this answer with the whole team.

---

## 7. How to present results that aren't final yet — without looking incomplete

We are mid-experiment. **Do not leave blank boxes and hope nobody notices.** Turn it into a
strength by showing the *evaluation framework* rather than a results table.

### The technique: present the method, mark the measurement

Build slide 4 around a **benchmark matrix** that shows what we measure and why, with status
markers instead of missing cells:

| Benchmark | What it tests | Status |
|---|---|---|
| ASVspoof 2019 LA (eval) | In-domain accuracy | `[[EER_INDOMAIN]]` |
| ASVspoof 2021 DF | Cross-dataset generalisation | `[[EER_DF21]]` (75 % coverage) |
| In-the-Wild | Real internet deepfakes, unseen tools | ⏳ measurement in progress |
| Indian-language corpora | Language robustness | ⏳ measurement in progress |
| Our own labelled recordings | Real-world false alarms | `[[FA_REAL]]` |

Then one honest caption underneath:

> *Evaluation is ongoing. Cross-dataset EER for audio anti-spoofing is typically 10–30 % —
> degradation is the known open problem in this field, and we measure it rather than avoid it.
> Final figures are reported against the locked model at submission.*

**Why this works.** It shows you know what the right benchmarks *are*, that you know what an
honest result looks like, and that you have not cherry-picked. A reviewer reading five decks
that all claim 99 % accuracy will notice the one that names its open problems. Stating the
10–30 % expectation *before* reporting a number also inoculates you — a reviewer cannot use it
against you if you raised it first.

**What not to do:** don't write "TBD" in a table cell, don't leave an empty chart, and don't
guess a number you intend to correct later. Either a real figure, a clearly-marked
placeholder, or an explicit "measurement in progress".

---

## 8. The visual specification — build these, not paragraphs

This is the section that matters most for the portal round. Each slide gets **one primary
visual**. Text supports the picture, never the other way round.

### Slide 1 — Title
No visual. Official SIH fields only: PS ID **SIH26104**, theme, category (Software), Team ID,
Team Name. Clean and correct.

### Slide 2 — Idea / Solution
**Primary visual: the attack-and-defence flow.** Horizontal, five steps, left to right:

```
[Scammer]──clones voice──▶[AI clone]──calls──▶[Victim's phone]
                                                     │
                                            ┌────────▼────────┐
                                            │   SONIX (live)   │
                                            │  🟢 🟡 🔴 band   │
                                            └────────┬────────┘
                                                     ▼
                                            [Human decides]
```
Use real icons, not ASCII. **Colour the band GREEN/AMBER/RED — that traffic light is our
product's identity and it should appear on at least three slides.**

Beside it, max 6 bullets: what it does, real-time not post-call, flags never blocks,
consent-gated, thresholds configurable per organisation.

### Slide 3 — Technical Approach
**Primary visual: the pipeline diagram.** Horizontal flow, each stage a labelled box with its
data shape underneath:

```
Audio 16 kHz → [4 s window, 0.5 s hop] → [Silence gate] →
[wav2vec2 XLS-R 300M ❄ FROZEN] → 1024-dim →
[MLP head 🔥 TRAINED · 262,657 params] → sigmoid →
[5-window average + hysteresis] → 🟢🟡🔴
```

**Use a snowflake on the frozen block and a flame on the trained one.** It reads instantly and
it visually makes our architectural argument (§4②) without a sentence of explanation.

**Secondary visual, if space: the windowing timeline.** Overlapping bars along a time axis
showing 4-second windows stepping every 0.5 s. It makes "real-time streaming" concrete rather
than asserted — and streaming is our headline differentiator.

Also list the stack in small type: Python, PyTorch, transformers, Streamlit, aiohttp
WebSockets, Chrome extension.

### Slide 4 — Feasibility & Viability
**This is our strongest slide. It gets two charts.**

**Chart A — the padding fix.** Line chart. X = clip length (1 s, 2 s, 3 s, full). Y = model's
fake-score, 0–1. Two lines: "before" and "after". A horizontal dashed line at 0.10 labelled
"Amber threshold". The before-line sits far above it; the after-line far below.
Caption: *"A 1-second clip of a real human went from confidently fake to confidently real."*
Y-axis label in plain words: **"model's fake score (higher = thinks it's fake)"**.

**Chart B — the shortcut.** Scatter plot. X = clip SNR in dB. Y = median score. Genuine clips
in one colour, synthetic in another. A vertical line at ~38 dB showing the split.
Caption: *"The model had learned 'clean background = fake', not synthesis detection. We found
it, proved it, and fixed it in the training data."*

Then the benchmark matrix from §7, plus a short risk/mitigation list.

### Slide 5 — Impact & Benefits
**Primary visual: a stakeholder map** — concentric or a simple 4-quadrant — showing
individuals, banks and call centres, telecom operators, government helplines, with the benefit
named for each.

Avoid a wall of statistics. If you cite fraud figures, cite the source on slide 6. Never use a
number you cannot attribute.

Emphasise: human-in-the-loop, per-organisation threshold tuning, and that the system is
language-aware by design after what we learned.

### Slide 6 — Research & References
Two columns: **Datasets** (ASVspoof 2019 LA, ASVspoof 2021 DF, In-the-Wild, IndicVoices,
MLAAD, IndicSynth) and **Papers** (Müller et al. INTERSPEECH 2022 on generalisation; the
wav2vec2/XLS-R paper; RawBoost augmentation; the MLAAD paper).

Note licence constraints where they apply (MLAAD is CC BY-NC — non-commercial).

### Design rules across every slide
- **Max 6 bullets. Minimum 14 pt.**
- One primary visual per slide; do not crowd two competing diagrams.
- Consistent colour: keep GREEN/AMBER/RED reserved for the risk band and never reuse those
  three as decorative colours.
- Every chart carries a caption stating its **conclusion**, not just what it plots.
- Readable with the sound off. That is the test.

---

## 9. The 6-minute demo — script it

Six minutes is longer than the slide time. The organisers are telling you they care more about
it working than about the pitch.

1. **(30 s)** Dashboard at rest. Point at the model selector and the threshold sliders —
   "configurable per organisation, live."
2. **(90 s)** Upload a genuine recording. Let the graph build window by window. Stays GREEN.
   Narrate what the line is.
3. **(90 s)** Upload a cloned recording **of the same speaker**. It climbs into RED. The
   matched pair is the moment that lands — same voice, opposite verdict.
4. **(60 s)** Switch models in the sidebar, re-run the same clip. Proves the architecture claim
   is real and not a slide.
5. **(60 s)** Show the consent gate and the audit trail.
6. **(30 s)** Buffer. Something always goes slightly wrong.

**Record a backup video of a clean run.** If the laptop dies or the model misbehaves, the video
*is* the demo. Winners' retrospectives are unanimous that teams who skip this regret it.

**Freeze all code changes two hours before.** After that, only testing.

---

## 10. Jury Q&A — the five minutes that decide it

**"Does it work on any audio?"** In-domain `[[EER_INDOMAIN]]`. Cross-dataset it degrades — the
acknowledged open problem in this field, and exactly why SONIX never blocks a call. Don't
oversell; the honesty plays better.

**"Can it detect a cloned Indian-language voice?"** We have proven we do not false-alarm on
genuine Indian speech and we catch Western synthetic speech. Detecting a cloned Hindi voice is
our next measurement — data is in hand, running now. **Do not claim it works.**

**"Why EER and not accuracy?"** Benchmark data is ~90 % spoof. A model that says "fake" every
time scores 90 % accuracy and is useless. EER is the field standard.

**"What's your false alarm rate on real calls?"** `[[FA_REAL]]`, and state the sample size
honestly. We found and fixed two separate causes — the padding bug and the noise shortcut.

**"What happens when a new cloning tool appears?"** Front-end frozen, only a 262k head is
trained. New attack family = extract embeddings once, retrain in minutes. Deliberate choice.

**"Privacy and consent?"** Pairing-code consent gate before any scoring, immutable per-call
audit trail, no audio retained past the call. Designed in, not bolted on.

**"Cost and scale?"** ⚠️ **Nobody has prepared this and it is a known differentiator.** Someone
must spend 30 minutes on: GPU inference cost per call-minute, concurrent calls per instance,
rough deployment cost. Past SIH winners cite financial literacy as what separated them.

**If you don't know, say so and say what you'd measure.** To a practitioner jury that reads as
competence. Bluffing does not.

---

## 11. Never say

- ❌ "99 % accurate" — we don't use accuracy and don't have that number
- ❌ "We fixed the Indian-speech problem, 57 % → 0.02 %" — see §6
- ❌ "Detects any AI voice" — it detects what it was trained against; we're explicit about gaps
- ❌ Any latency figure nobody has measured
- ❌ Any number not in the LOCKED table or supplied by Yugal

---

## 12. Sources

- SIH 2026 PPT template & evaluator scoring — https://blogs.reskilll.com/sih-2026-ppt-template-exact-format-slides-evaluators-score/
- SIH idea presentation format — https://www.lets-code.co.in/blogs/sih-2025-complete-guide-ppt-template/
- SIH 2022 winner's guide — https://medium.com/@arinjay_11020/how-i-won-smart-india-hackathon-2022-a-step-by-step-guide-by-arinjay-pathak-501529c00abb
- SIH 2024 winners' retrospective — https://how-we-won-sih-24-and-survived-it.hashnode.dev/everything-about-winning-sih-2024
- Müller et al., "Does Audio Deepfake Detection Generalize?" — https://arxiv.org/pdf/2203.16263

---

## 13. Final checklist

- [ ] Official SIH template confirmed from the portal (Yugal) and followed exactly
- [ ] Every slide readable **with the sound off** — the portal round has no presenter
- [ ] Max 6 bullets per slide, minimum 14 pt
- [ ] One primary visual per slide, built as specified in §8
- [ ] Slide 4 has both charts: padding fix + SNR shortcut
- [ ] "Higher score = model thinks it's fake" stated wherever scores appear
- [ ] Every DF21 figure carries "(75 % key coverage)"
- [ ] Unfinished results shown as a benchmark matrix with status, not blank cells
- [ ] Every `[[PLACEHOLDER]]` listed in a table at the end with an owner
- [ ] Green/Amber/Red used only for the risk band, never as decoration
- [ ] Demo script rehearsed; backup video recorded
- [ ] Cost/scale answer prepared
- [ ] Indian-clone Q&A answer agreed by the entire team
