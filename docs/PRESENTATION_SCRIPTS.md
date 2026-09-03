# SONIX — Presentation & Prototype Scripts

**SIH 2026 · SIH26104 · Team SONIX**
Companion to `docs/SONIX_MASTER.md`. That file is the facts; this file is the words.

**Read before rehearsing:**

1. **Never say a number that is not 🔒 in `SONIX_MASTER.md` §6.** Every figure in
   these scripts is checked against that ledger. `[[LIKE_THIS]]` means *do not
   fill it in from memory* — the owner fills it in or the line gets cut.
2. **Higher score = the model thinks it is FAKE.** Say it out loud the first time
   a score appears on screen. Judges misread the axis otherwise.
3. **We do not claim a best model.** We show what each one buys and what it costs.
4. **Everything is timed.** Read at ~135 words per minute. If you are over, cut a
   sentence — do not speed up.

---

## Part 0 — Running order at a glance

### Slides — 4 minutes

| Slide | Speaker | Time | Cumulative |
|---|---|---|---|
| 1 · Cold open + Title | **Suryansh** | 0:35 | 0:35 |
| 2 · Idea / Solution | **Suryansh** | 0:35 | 1:10 |
| 3 · Technical Approach | **Yugal** → **Akshat** | 1:05 | 2:15 |
| 4 · Feasibility & Viability | **Yukti** | 0:50 | 3:05 |
| 5 · Impact & Benefits | **Navya** | 0:35 | 3:40 |
| 6 · Research & Datasets | **Anubhav** | 0:20 | 4:00 |

### Prototype — 6 minutes

| Segment | Speaker | Time | Cumulative |
|---|---|---|---|
| A · The models we built, and their limits | **Yugal** | 1:30 | 1:30 |
| B · The datasets, one line each | **Yugal** | 0:45 | 2:15 |
| C · Live test — genuine, then the clone | **Suryansh** | 1:30 | 3:45 |
| D · What is happening under the hood | **Akshat** | 1:00 | 4:45 |
| E · Swapping models live + consent/audit | **Yugal** → **Suryansh** | 0:50 | 5:35 |
| F · Buffer | — | 0:25 | 6:00 |

**Handover rule:** the person finishing says the next person's *name*.
"Yugal will take the architecture." It stops the two-second dead air that makes a
team look unrehearsed.

---

# PART 1 — THE SLIDE PRESENTATION (4 minutes)

---

## SLIDE 1 — Cold open + Title · **SURYANSH** · 0:35

### The opening — do this, it is worth more than any slide

**Preferred version (needs 30 minutes of prep before the round):**

Clone Suryansh's own voice with one of the free tools we already used for the
test set. Record it saying one plain sentence. Have it ready on the laptop.

> **[Do not introduce it. Do not say a word. Press play.]**
>
> **[CLONED AUDIO, ~6 seconds]:**
> *"Good morning. I'm Suryansh from Team SONIX, and everything I am about to
> tell you is true."*
>
> **[Stop the audio. Wait one full second. Then speak live, same voice, same
> room:]**
>
> "That wasn't me.
>
> That was ten seconds of my voice, a free website, and about four minutes of
> work. Nobody in this room could tell — and neither could my bank."

**[Beat. Now the title slide.]**

> "We are **Team SONIX**, and we're here for problem statement **SIH26104** from
> the AICTE Cyber Security Cell — **real-time detection of AI-cloned voices in
> live calls.**"

**Fallback if the clone isn't ready** — still strong, no props:

> "In the time it takes me to say this sentence, there is enough of my voice on
> this recording to clone it. Not in a lab — on a free website, in about four
> minutes.
>
> That is the world our banks, our helplines and our parents are already living
> in. **We are Team SONIX**, and this is problem statement **SIH26104** — real-time
> detection of AI-cloned voices in live calls."

**Stage directions**
- Do not stand behind the laptop. Step forward before you press play.
- The one-second silence after "That wasn't me" does more work than any slide.
- Do not smile through the cold open. Deliver it flat. The joke, if any, comes later.
- If a judge laughs or reacts — let them. Do not talk over it.

---

## SLIDE 2 — Idea / Solution · **SURYANSH** · 0:35

**Visual on screen:** the attack-and-defence flow (flowchart A, `SONIX_MASTER.md` §14).

> "Here's how the attack actually runs. A scammer takes seconds of public audio —
> a YouTube clip, a voice note, an old interview. A free tool clones it. The
> victim's phone rings, and the voice on the other end is somebody they trust:
> a bank official, a CEO, a family member. The money moves before anyone doubts
> the voice.
>
> Every detection tool we found analyses a recording *afterwards*. That's an
> autopsy, not a defence.
>
> **SONIX scores the call while it is happening.** Green, amber, red — updated
> twice a second, on live audio.
>
> And one design decision we'd defend anywhere: **SONIX never blocks a call.** A
> false positive that cuts off a real emergency is a worse failure than the fraud
> we're preventing. We flag. A human decides.
>
> Yugal will take you through how it works."

**Bullets on the slide (max 6, do not read them aloud):**
real-time, not post-call · flags, never blocks · consent-gated · thresholds
configurable per organisation · runs inside the bank's own network · updates
twice a second

---

## SLIDE 3 — Technical Approach · **YUGAL** (0:40) → **AKSHAT** (0:25) · 1:05

**Visual:** pipeline diagram (flowchart B) left, integration diagram (flowchart C)
right, SDK snippet below.

### YUGAL — 0:40

> "Audio comes in at 16 kilohertz. We cut it into **four-second windows every half
> second** — 87 percent overlap, so the verdict refreshes twice a second. Windows
> that are near-silent get dropped before they're scored.
>
> Each window goes through **wav2vec2 XLS-R — 300 million parameters — and we keep
> it frozen.** On top of it sits a small trained head: **262,657 parameters.**
>
> That split is the architecture decision I'd most want you to take away. The
> frozen part does the listening; the tiny trained part does the deciding. So
> when a new cloning tool appears next month, we extract embeddings once and
> **retrain in minutes** — not days, and with no redeployment. Our API already
> serves several trained heads behind one endpoint, sharing one copy of the
> front-end, with no extra GPU memory.
>
> The raw output is smoothed across five windows, with hysteresis — three of five
> have to agree before the band changes — so one strange window can't flip a
> verdict."

### AKSHAT — 0:25

> "And the thing the problem statement asks for three times is an **API and SDK** —
> not an application. So that's what we built.
>
> SONIX is middleware. **Seven REST endpoints and a WebSocket stream.** Three lines
> of code and a bank's existing system holds a transfer on a red band.
>
> The evidence that it's a real interface and not just our app's backend:
> **two completely independent clients already run on it** — a Python dashboard
> and a JavaScript Chrome extension that share no code between them.
>
> Yukti will show you how we test it."

---

## SLIDE 4 — Feasibility & Viability · **YUKTI** · 0:50

**Visual:** Chart A (padding fix) and Chart B (SNR shortcut) side by side, then the
benchmark matrix.

> "I'll show you two failures we found in our own model, because how a team
> handles its own bugs says more than any accuracy number.
>
> Our detector was calling real people fake. We didn't touch the threshold — we
> asked why. Short clips were being padded with **digital silence**, so a
> one-second clip was three-quarters nothing. A real human at one second scored
> **0.88** — confidently fake. We changed the padding to repeat the audio instead.
> Same clip, same model: **0.03**. Confidently real.
>
> **[Point at Chart B.]**
>
> Genuine recordings still failed. So we plotted score against
> signal-to-noise ratio — and found the model had learned **'clean background
> equals fake'**, not synthesis detection. Our training data's silence sits at
> minus 75 dBFS, a floor no real microphone produces, and phone noise-suppression
> strips room tone out of genuine calls. Our failing genuine clips sat between
> **39 and 67 decibels**; the ones that passed sat at **35 to 38**.
>
> We fixed it in the **training data**, not the thresholds — real room reverb and
> real noise, so background can't be a cue any more.
>
> **[Point at the matrix.]**
>
> This is our evaluation framework, with honest status. Cross-dataset error rates
> in this field typically land between **10 and 30 percent** — generalisation is
> the known open problem, and we measure it rather than avoid it.
>
> Navya will tell you who this is for."

**⚠️ Yukti — hard rules for this slide**
- Say **"score"**, never "accuracy". If asked, the metric is **EER**, because the
  benchmark data is ~90 % spoof and accuracy would be flattering and meaningless.
- Any DF21 figure must carry **"at 75 percent key coverage"**.
- If a number in the matrix is still `[[PENDING]]`, say *"in progress"* and move on.
  Do not estimate. Do not round up a dev number into a headline.

---

## SLIDE 5 — Impact & Benefits · **NAVYA** · 0:35

**Visual:** stakeholder map (flowchart E).

> "Four groups, and they need different things from the same score.
>
> **Individuals** — the family-emergency scam, the fake call from a relative in
> trouble. **Banks and contact centres** — where a red band can hold a transfer
> before it clears, which is the only moment where stopping fraud is still cheap.
> **Telecom operators**, who could run this at network level and offer it as a
> service. And **government helplines**, where official impersonation is the whole
> attack.
>
> The thresholds are theirs to set. A bank would run this stricter on a fund
> transfer than on a general enquiry — same model, different operating point.
>
> And on privacy, which we designed in rather than bolted on: there is a
> **consent gate enforced in the state machine** — audio is dropped, in code,
> unless the call is approved. Every call writes an **immutable audit trail**.
> **No audio is retained** — only the score.
>
> Anubhav has our datasets and references."

---

## SLIDE 6 — Research & Datasets · **ANUBHAV** · 0:20

**Visual:** two columns — datasets left, papers right.

> "Everything we trained and tested on is public and standard: **ASVspoof 2019,
> 2021 DeepFake and ASVspoof 5** for synthetic speech, **In-the-Wild** for real
> deepfakes off the internet, and for Indian languages, **IndicVoices** for genuine
> speech and **IndicSynth** plus **Meta's MMS-TTS** for synthetic.
>
> On top of that we recorded and labelled our **own** real and cloned clips,
> because none of those benchmarks contains an Indian public figure cloned by the
> tools a scammer would actually use.
>
> Methods are from Müller and colleagues at INTERSPEECH, the wav2vec2 and XLS-R
> papers, and RawBoost.
>
> **[Hand over]** — "Let's show you the thing running."

---

# PART 2 — THE PROTOTYPE (6 minutes)

> **Before the round:** freeze the code two hours ahead. Set `HF_HUB_OFFLINE=1`.
> Start the server and run one full clip through it so the model is warm.
> **Record a backup video of a clean run** — if the laptop dies, the video is the demo.

---

## SEGMENT A — The models we built, and their limits · **YUGAL** · 1:30

**On screen:** the dashboard sidebar, model list visible.

> "Before I run anything, I want to be straight with you about what's in this
> sidebar — because there are **seven trained models here, and I'm not going to
> tell you one of them is the best.** We don't have the evidence to say that yet,
> and I'd rather show you what each one bought us and what it cost.
>
> They all share the same frozen front-end. What changes is what each head was
> trained on.
>
> **The baseline** is trained on clean benchmark speech only. It's our strongest
> in-domain reference point — and it's also the one that taught us both of our
> biggest lessons, because it fell apart on phone audio and on real recordings.
>
> **The codec model** adds phone-compressed copies of everything. It holds up on
> compressed call audio. Its number looks better than the baseline's, but it's
> measured on a *different test set*, so those two are not comparable and we don't
> present them side by side.
>
> **The robust model** adds simulated microphones and channel noise, and then
> **room reverb and background noise** on top. That's the version where background
> can no longer be a shortcut. But it had never heard an Indian voice — and it
> flags genuine Indian speakers as fake.
>
> **So we added Indian speech.** The false alarms collapsed. And then we tested a
> **matched pair** — a real recording and a cloned recording of the *same speaker* —
> and it passed both as genuine. It hadn't learned to detect anything. It had
> learned **'Indian audio is real'**, because every Indian clip it had ever seen was
> genuine. That near-zero false-alarm number is honest, and it measures nothing
> about detection. We're not putting it on a slide.
>
> **This model** — the one I'll run in a moment — is the first one with Indian
> languages on **both** sides of the label: 35,200 synthetic Indian clips across
> three different generator families and twelve languages. Whether that actually
> fixed it is **the measurement we are running right now**, and I'll tell you the
> answer either way.
>
> Two limits I want on the record. **One: days, not weeks.** These heads were
> trained in single sessions on consumer GPUs. No hyperparameter search, no
> ensembling, no cross-validation. **Two: our data.** The public benchmark is
> studio-clean, the Indian genuine-speech corpus contains no fakes at all, and
> one dataset we wanted is still behind an approval wall. Every one of those
> limits shows up somewhere in these seven models."

**⚠️ Yugal — do not say**
- "This is our best model" · any EER figure · "we fixed Indian speech" ·
  any latency number · "production-ready"

---

## SEGMENT B — The datasets, one line each · **YUGAL** · 0:45

> "One line on each, and why it's in here.
>
> **ASVspoof 2019 LA** — the field's standard benchmark of synthetic and converted
> speech. It's what 'synthetic' *means* to our baseline.
>
> **ASVspoof 2021 DeepFake** — a harder, compressed track. We never train on it;
> it's purely how we measure whether we generalise.
>
> **ASVspoof 5** — the newest edition, with modern generators. It stops us being
> stuck on 2019-era attacks.
>
> **In-the-Wild** — real deepfakes scraped off the internet, made with tools nobody
> told us about. The closest thing to a field test that exists.
>
> **IndicVoices** — genuine Indian multilingual speech. It's what stopped us
> false-alarming on Indian accents — and because it is **genuine-only**, it is also
> exactly what caused the second problem I just described.
>
> **IndicSynth** — Indian-language synthetic speech across twelve languages. The
> missing half of that label.
>
> **MMS-TTS** — Meta's multilingual text-to-speech. We generated our own Hindi,
> Marathi, Bengali, Tamil and Telugu spoofs with it, so our Indian fakes come from
> **two independent generator families**, not one.
>
> **MUSAN and room impulse responses** — not speech data at all. Noise and reverb.
> They exist purely to stop the model using a clean background as a shortcut.
>
> **G.711 and RawBoost** — telephony codec and channel augmentation, so phone audio
> is in-domain instead of a surprise.
>
> **And our own recordings** — real and cloned clips we made and labelled ourselves,
> because no public benchmark contains an Indian public figure cloned with the free
> tools a scammer would actually reach for.
>
> **MLAAD** we wanted and don't have — it's gated behind manual author approval and
> we're still waiting. I'd rather tell you that than imply we used it.
>
> Suryansh, run it."

---

## SEGMENT C — Live test · **SURYANSH** · 1:30

**⚠️ Rehearse this with the actual clips and pick the pair that works.** Use a
**matched pair from `sonix_real/`** — the same speaker, real and cloned. A matched
pair is worth ten unmatched ones.

**0:00–0:45 — the genuine clip**

> "This is a genuine recording. Nothing about it is precomputed — the model is
> consuming it four seconds at a time, exactly the way it would consume a live
> call.
>
> **[Upload. Let the graph start.]**
>
> One thing to read correctly: **higher on this axis means the model thinks it's
> fake.** So flat and low is good. Amber sits at 0.10, red at 0.90 — and those are
> the operator's to set.
>
> **[Let it run. Don't fill the silence with talking.]**
>
> Green. That line is 40-odd separate decisions, half a second apart."

**0:45–1:30 — the clone. This is the moment.**

> "Now the same speaker — cloned.
>
> **[Upload. Then say nothing. Let the room watch the line move.]**
>
> **[Only after it has clearly moved:]** Same speaker, same length, same recording
> conditions. The only thing that changed is that a machine said it instead of a
> person."

**If the clone does NOT flag — say this, immediately, do not stall:**

> "And that's a miss — I'll show you exactly that rather than hide it. This is a
> generator our training data has never seen, and it is precisely the
> generalisation gap Yukti described. It's why this system flags for a human
> instead of blocking a call on its own."

**A calm, honest miss costs you far less than a visible scramble.** Practise
saying it out loud so it doesn't sound like an excuse.

---

## SEGMENT D — What is happening under the hood · **AKSHAT** · 1:00

**On screen:** the telemetry panel — per-window scores, VAD decisions, ring-buffer stats.

> "Let me show you what the dashboard is actually reading, because this is the
> part that makes it a live system rather than a file classifier.
>
> **[Point at the window timeline.]**
>
> Every point on that line is one four-second window, and they overlap by 87
> percent — so a verdict updates twice a second instead of once a call.
>
> **[Point at the greyed-out windows.]**
>
> These greyed windows were **dropped by the silence gate** before they ever
> reached the model. That matters: a window that's mostly silence looks like
> nothing the model was trained on, and it tends to guess 'fake'. Dropping them
> removed a large share of our false alarms, and it costs microseconds.
>
> **[Point at the band, not the raw line.]**
>
> The band isn't the raw score. It's a **five-window moving average with
> hysteresis** — three of five windows have to agree before the colour changes.
> So a single strange window can't turn a real call red.
>
> Two engineering points I'd want a reviewer to know. **First**, the scoring
> engine batches windows across *concurrent* calls through one shared front-end —
> that's how one GPU serves several calls without loading the model more than
> once. **Second**, this whole panel exists because we needed to be able to prove
> the silence gate does what we say it does, rather than ask you to take our word
> for it. Every window is logged whether it passed or not.
>
> Yugal — swap the model."

---

## SEGMENT E — Model swap + consent and audit · **YUGAL** (0:30) → **SURYANSH** (0:20)

### YUGAL — 0:30

> "Same audio. Different trained head.
>
> **[Switch model in the sidebar. Re-run the same clip.]**
>
> No reload, no second GPU, no redeployment — the 300-million-parameter front-end
> stayed exactly where it was, and only the 262,000-parameter decision layer
> changed.
>
> **[Point at the difference in the two lines.]**
>
> That's the answer to 'what happens when a new cloning tool appears in March'.
> We extract embeddings once, retrain a head in minutes, and serve it alongside
> the old one so they can be compared. The old model doesn't go away."

### SURYANSH — 0:20

> "Last thing, and it's the one we'd defend hardest.
>
> **[Show the consent gate, then the audit JSON.]**
>
> Audio is dropped **in the state machine** unless the call is approved — that's
> enforced in code, not in a policy document. Every call writes an immutable audit
> record: what was scored, when, under which model. And **no audio is retained** —
> only the score.
>
> That was in the first version we built, not added after someone asked."

---

## SEGMENT F — Buffer · 0:25

Do not fill it. If you finish early, stop and say:
> "That's the prototype — happy to take questions."

Silence at the end reads as confidence. Rambling reads as padding.

---

# PART 3 — Rehearsal checklist

**Two days before**
- [ ] Every speaker has read `SONIX_MASTER.md` §6 (the numbers ledger)
- [ ] Suryansh's cloned-voice cold open recorded, or the fallback memorised
- [ ] The `sonix_real/` matched pair chosen and tested end-to-end
- [ ] Every `[[PLACEHOLDER]]` either filled by its owner or cut from the script

**One day before**
- [ ] Full run-through, timed, standing up, in order, no notes
- [ ] **Backup demo video recorded** from a clean run
- [ ] Everyone knows their handover line and the next person's name
- [ ] The "it didn't flag" line rehearsed out loud by Suryansh

**Two hours before**
- [ ] Code frozen — no commits after this
- [ ] `HF_HUB_OFFLINE=1` set; server started; one clip run through to warm the model
- [ ] Laptop on mains power, notifications off, second screen tested
- [ ] Clips folder open, files renamed so the right one is obvious under pressure

**On the day — the three sentences that save you**
- Wrong number in your head → *"I'd rather check that than guess — it's in our results table."*
- Demo fails → *"That's the live path failing; here's the recorded run, and here's what's happening underneath."*
- Question you can't answer → *"We haven't measured that. Here's how we would."*

To a practitioner jury, all three read as competence.
