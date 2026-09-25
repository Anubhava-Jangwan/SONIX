# SONIX: Project Context

Give this file to Claude along with the video brief. If anything in the script needs to change, Claude
should use only the facts in this file.

---

## 1. Basics

- **Competition:** Smart India Hackathon 2026
- **Problem statement:** SIH26104 (AICTE Cyber Security Cell): *AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks*
- **Theme / category:** Blockchain & Cybersecurity · Software
- **Team:** SONIX · UPES Dehradun
- **Members:** Yugal Mahajan (ML lead) · Navya · Akshat · Anubhav · Suryansh · Yukti

---

## 2. The problem

- Voice-cloning tools can copy a voice from a few seconds of audio.
- Scammers call banks, contact centres, companies and families pretending to be someone trusted.
- The person on the call cannot hear the difference, and nothing on the line checks for it.
- Sourced numbers you may use:
  - INR 22,845 crore lost to cyber fraud in India in 2024, 3× 2023 (MHA, Lok Sabha, 22 Jul 2025)
  - 47% of Indian adults have faced or know a victim of an AI voice scam; 83% of Indian victims lost money (McAfee, 2023)
  - 69% of Indians aren't confident they can tell a cloned voice (McAfee, 2023)
  - Deepfake call attempts rose 1,300%+ in 2024 (Pindrop, 2025)

---

## 3. What SONIX does

- Listens to a **live call** and flags if the voice is likely **AI-cloned**.
- Scores the audio **every 0.5 s** using a **4 s window**, so the first result comes about 4 s into speech.
- Shows a **Green / Amber / Red** band.
- **Never blocks or cuts a call.** It warns, and a person decides (for example, calls back on a known number).

### Where it works
- **Google Meet / web calls:** through the Chrome extension
- **Contact-centre / VoIP calls:** through the API
- **NOT ordinary mobile phone calls.** Android and iOS don't let apps read call audio. Never claim this.

---

## 4. How it works (simple)

1. **Consent gate:** nothing is analysed until an operator approves a 6-digit pairing code.
2. **Windowing:** the last 4 s of audio, checked every 0.5 s.
3. **Silence gate:** silent windows are skipped, so silence can't cause an alarm.
4. **Speech model:** a large pre-trained speech model (XLS-R, 300M parameters) that stays frozen.
5. **Detector head:** a small trained model that outputs the probability that the voice is synthetic.
6. **Smoothing:** a 5-window average plus a 3-of-5 rule, so the band doesn't flicker.
7. **Band:** Green / Amber / Red shown to the user.
8. **Audit record:** scores and events are saved; **audio is never stored**.

**Swappable heads:** many detection models share one speech model. When a new cloning tool appears, a
new small model can be trained in minutes and added **without changing the service**.

---

## 5. The two clients

### Chrome extension (Google Meet)
- Captures the **Meet tab's audio**, meaning the other participants, not your own mic.
- Shows the band in the popup and as an **overlay on the Meet page**.
- Start monitoring → pairing code → approve in the console → live band.

### Streamlit operator console
- Live **risk timeline** per call, with the green/amber/red bands.
- Pairing-code **approval** (consent).
- Telemetry and silence-gate view.
- Upload a recording and **compare several detection models side by side**.

Both use one SONIX service (REST + WebSocket).

---

## 6. What makes SONIX different (safe to say)

- **Built for Indian speech:** trained on real and fake speech in Indian languages plus English, not just English benchmarks.
- **We measured why detectors fail on Indian speech:** the standard speech representation recognises the *language* with 99.33% accuracy, and an English-trained detector scores at chance on Indian speech (AUC 0.52). We rebuilt the model around that finding.
- **Trained for real calls:** telephone and VoIP codecs (G.711, AMR-WB, Opus), room echo and background noise, applied equally to real and fake audio.
- **Passive:** works on any voice and doesn't need the cloning tool to add a watermark.
- **Human in the loop:** flags, never blocks.
- **Privacy:** consent first, no audio stored.

---

## 7. Datasets used

- **Real speech:** ASVspoof 2019 LA · IndicVoices (Indian languages) · LibriSeVoc
- **Fake speech:** ASVspoof 2019 LA attacks · MLAAD (19 TTS models) · IndicSynth (voice conversion) · MMS-TTS · LibriSeVoc (6 neural vocoders)
- **Test only:** In-the-Wild · ASVspoof 2019 LA eval · unseen Indian speakers · matched real-vs-clone recordings

---

## 8. Numbers allowed in the video / script

| Fact | Value |
|---|---|
| Check interval | every 0.5 s |
| Window | 4 s |
| Speech model | XLS-R, 300M parameters, frozen |
| Language recognised from the old representation | 99.33% |
| English detector on Indian speech | AUC 0.52 (0.5 = chance) |
| Best earlier model on matched real-vs-clone recordings | AUC 0.764 (n = 21, 9 true pairs) |

**Do not use any other numbers.** Specifically, never use accuracy %, EER %, latency, or any hardware/GPU names.

---

## 9. Never say

- "99% accurate", or any accuracy figure
- "Detects any AI voice" (say *flags likely AI-cloned voices*)
- "Works on phone calls" or "phone app"
- "Blocks the call" / "stops the scammer automatically"
- Any latency or speed number beyond "every 0.5 s"
- GPU / hardware details

---

## 10. Demo facts (for the video)

- In the Meet demo, the other participant plays one **real** clip (band stays **green**) and one **AI-cloned** clip (band turns **red**).
- The band needs about **4 s** after speech starts to show the first result.
- Meet's own **noise cancellation should be off**, because it changes the audio.
- The console shows the **same call as a timeline** afterwards.
