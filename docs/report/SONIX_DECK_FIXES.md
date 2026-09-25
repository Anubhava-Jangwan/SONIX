# SONIX deck: fixes after reviewing the current PDF

For Suryansh. Two ready-made images come with this file: `impact_stats.png` (4 stat tiles) and
`impact_fraud_chain.png` (diagram). Drop them into Canva as they are.

## The main problem: every slide tells the same story

"Flag, not block", the confound table and the pipeline diagram each appear 2–3 times. Give every
slide ONE job, and cut anything that belongs to another slide:

| Slide | Its one job | Question the judge should have answered |
|---|---|---|
| 2 Solution | what SONIX is + why now | "What is it and who needs it?" |
| 3 Technical | how the model works and why it's different | "Is this real engineering?" |
| 4 Feasibility | what's built, how we test it, what can go wrong | "Can they actually deliver?" |
| 5 Impact | who is protected, how much is at stake, where SONIX breaks the fraud | "Does this matter at national scale?" |
| 6 References | what we built on + what we add | "Is it grounded?" |

---

## Errors to fix first (these can cost credibility)

1. **Slide 2, dark screenshot: "EVAL EER 1.49%".** Remove it. That number is unverified, so it's on
   our never-use list.
2. **Slide 2, same panel: "LATENCY 0.5 s".** That's the *hop*, not latency. Latency hasn't been
   measured. Relabel it "SCORES EVERY 0.5 s" or delete it.
3. **Slide 2 chart (India vs Global).** It has no source on the slide. Add under it: *"Source: McAfee,
   Beware the Artificial Impostor (2023), n = 7,054 incl. 1,010 India."* Delete the "≈202 / ≈705"
   labels; they read as unexplained numbers.
4. **Slide 4 heading "WHAT IS COSTS TO RUN"** → "WHAT IT COSTS TO RUN". The "model stats"
   filler lines look unfinished. Replace them with `Latency · cost per call-minute · GPU needs:
   measured on final model`, or cut the box.
5. **Slide 4: "Handles silence, short clips, noisy and compressed audio".** This isn't measured yet.
   Change it to: "Trained with telephone codecs (G.711, AMR-WB, Opus), room echo and noise, applied
   equally to real and fake audio."
6. **Slides 5/table: "12 languages".** The v4 model trains on **8 Indian languages + English**.
   "12" was the old fake corpus. Use 8.
7. **Slide 5: "first balanced Indic corpus".** Nobody can prove "first", and a judge can challenge it.
   Say "a balanced Indic real-vs-fake training set".
8. **Slides 1 and 6 are empty.** Fill them (below).

---

## Slide 5: Impact & Benefits (full redesign)

**Title bar:** IMPACT AND BENEFITS
**Headline (big, one line):**
> **₹22,845 crore was lost to cyber fraud in 2024. A cloned voice leaves no trace a human can hear.**

**Row 1: `impact_stats.png`** (4 tiles, sources printed on each tile):
- ₹22,845 cr lost in 2024, 3× 2023, 36.4 lakh incidents (MHA, Lok Sabha, 22 Jul 2025)
- 47% of Indians hit or know a victim of an AI voice scam; 83% of victims lost money (McAfee 2023)
- +1,300% deepfake call attempts in 2024; banks +149% (Pindrop 2025)
- ₹5,489 cr saved by acting fast via CFCFRMS: speed recovers money (MHA, 22 Jul 2025)

**Row 2: `impact_fraud_chain.png`**: the 5-step fraud call (Harvest → Clone → Call → Pressure →
Money moves) with SONIX cutting in between steps 4 and 5. This is the slide's diagram and replaces
the text boxes.

**Row 3: three short columns, "Who is protected", one concrete benefit each:**

| Banks & fintech contact centres | Enterprises on Meet / web calls | Cyber cells & investigators |
|---|---|---|
| Step-up checks only on flagged calls, so genuine customers aren't slowed down | "CEO / CFO voice" payment approvals get a second, machine listener | Score a recorded call after the fact to prioritise clone-voice complaints |

**Small footer line:** One API serves many institutions, so every bank doesn't have to build its own
detector.

**Remove from this slide:** the "Design Stance" box (already on Slide 2) and the "Research
Contribution" box (moves to Slide 6). Remove the "SONIX vs existing tools" table too: "Existing tools:
full retrain / auto-block / standalone" is unsourced, and a judge will ask which tools.
If you want a comparison, use only this defensible row: **Watermarking (e.g. AudioSeal) only
catches voices from generators that cooperate; SONIX is passive and listens to any voice.**

---

## Slide 2: Proposed Solution (tighten)

- Keep the pipeline diagram (Listens → Scores → Flags). It's good.
- Fix errors 1–3 above.
- Merge "Why this solution" and "Why existing detectors fail" into one box, **"Why current
  detectors miss Indian clones"**, with one line: *English-trained detectors score AUC 0.52 on
  Indian speech (chance is 0.50); we measured why and rebuilt for it (Slide 3).* That points the judge
  to Slide 3 instead of repeating it.

## Slide 3: Technical Approach (currently missing the actual model)

Right now it shows the confound table and a UI screenshot, but **not how v4 works**. Move the UI
screenshot to Slide 4 and put this on the right half, as 4 boxes, each tied to a measured problem:

| Measured problem | v4 fix |
|---|---|
| Embedding keeps language, loses clone artifacts | Listen to layers 5, 6, 24 with mean + std pooling (layer 6 best on unseen attacks: 0.9842 vs 0.9532) |
| "Indian = real" shortcut | Real *and* fake audio in all 8 Indian languages: IndicVoices 36,859 real · IndicSynth 16,800 · MLAAD 19,000 (19 TTS models) · LibriSeVoc 13,201 real + 25,200 fake (6 vocoders) · ASVspoof 25,380 |
| Model can quietly fail one language | Loss balanced over language × generator × real/fake; checkpoint chosen on the **worst** group |
| Phone codecs hide artifacts | Every training clip, real and fake, also passed through G.711 / AMR-WB / Opus + echo / noise / RawBoost |

Bottom strip, "We audit our own shortcuts": silence-padding bias found and fixed (short clips were
71% of ASVspoof vs 3.8% of IndicSynth) · 682 leaked speakers removed from the test set; the headline
test uses **256 unseen Indian speakers**.

## Slide 4: Feasibility (make it "can they deliver")

- Put the UI screenshot here as **"Built and running today"** (console + Chrome extension, 7 REST
  endpoints + WebSocket, consent gate).
- **How we'll prove it (evaluation plan):** In-the-Wild 31,779 clips · ASVspoof eval 71,237 · 256
  unseen Indian speakers (2,551 clips) · matched real-vs-clone pairs (n = 21, 9 true pairs, now
  being scaled to 200+ volunteers). Results: `[[PLACEHOLDER: after v4]]`.
- Keep the head-comparison bars (they're good). Keep "n = 21, 9 true pairs" next to them.
- Keep the Risks + Mitigation list, and remove the duplicate pipeline diagram (it's on Slide 2).

## Slide 1: Title (fill the template fields)

- Problem Statement ID: **SIH26104**
- Problem Statement Title: *AI-Powered Real-Time Detection and Prevention of Voice Cloning
  Impersonation Attacks*
- Theme: **Blockchain & Cybersecurity**
- PS Category: **Software**
- Team ID: **SIH-UPES-2026-T302**
- Team Name: **SONIX**

## Slide 6: Research & References

**What we add (3 bullets):**
1. A measured diagnosis: the standard speech embedding identifies language at 99.33% and transfers
   English→Indian at AUC 0.52 (chance).
2. A training recipe built on it: multi-layer pooling, language × generator balancing, worst-group
   selection, and symmetric channel augmentation.
3. A working real-time, human-in-the-loop service with swappable detector heads.

**References:**
1. Sun, Yang & Lyu (2026). A survey of AI-generated voices and their detection. APSIPA Trans. SIP 15(1), 76–109. DOI 10.1108/ATSIP-09-2025-0093
2. San Roman et al. (2024). Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal). ICML 2024. arXiv:2401.17264
3. Bird & Lotfi (2023). Real-time Detection of AI-Generated Speech for DeepFake Voice Conversion. arXiv:2308.12734
4. McAfee (2023). Beware the Artificial Impostor. mcafee.com
5. Ministry of Home Affairs, Lok Sabha reply on cyber-fraud losses, 22 Jul 2025
6. Pindrop (2025). Voice Intelligence & Security Report
7. Datasets: ASVspoof 2019 LA · IndicVoices · IndicSynth · MLAAD · LibriSeVoc · MMS-TTS · In-the-Wild
