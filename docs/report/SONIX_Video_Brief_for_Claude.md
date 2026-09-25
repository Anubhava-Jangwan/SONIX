# Brief for Claude: SONIX prototype video (2 minutes)

Paste this whole file into Claude. It has everything needed to help produce the video.

---

## What SONIX is

SONIX (Team SONIX, Smart India Hackathon 2026, problem statement SIH26104) detects AI-cloned voices
during live calls. It scores the call audio every 0.5 s on a 4 s window and shows a
**Green / Amber / Red** band. It **never blocks or cuts a call**. It flags, and a person decides.

Two clients use one SONIX service:
- a **Chrome extension** that monitors a Google Meet tab
- a **Streamlit operator console** with a live risk timeline and side-by-side comparison of detection models

---

## The video

- **Length:** 2 minutes
- **Part 1 (0:00–1:02):** a live Google Meet call. The other participant plays one **real** voice clip
  (band stays green) and one **AI-cloned** clip (band turns red), while the SONIX extension shows the result.
- **Part 2 (1:02–2:00):** switch to the Streamlit console: the same call's timeline, then several models
  compared on the fake clip.

---

## Script

| Time | Screen / action | Voice-over |
|---|---|---|
| 0:00 | Meet call running, extension visible | This is SONIX, live on a Google Meet call. It listens to the other person and flags if their voice is AI-cloned. |
| 0:08 | Click SONIX icon → Start monitoring → approve | We turn on monitoring. From here, SONIX checks the call every half second. |
| 0:15 | Other person plays **REAL** clip · band stays **GREEN** | First, a real human voice. The band stays green, so nothing suspicious. |
| 0:35 | Other person plays **FAKE** clip · band goes **AMBER → RED** | Now an AI-cloned voice. Within a few seconds, SONIX turns red: likely synthetic. |
| 0:55 | Hold on the red overlay | It doesn't cut the call. It warns you, and you decide. |
| 1:02 | Switch to Streamlit console | Everything you just saw runs through the SONIX service. Here's the operator console. |
| 1:05 | This call's timeline | This is the same call as a timeline: green while the real voice spoke, red when the clone started. |
| 1:25 | Upload fake clip → run several models side by side | We can also test any recording with several detection models side by side. When a new cloning tool appears, we add a new model without touching the service. |
| 1:45 | SONIX logo + team name | SONIX: real-time AI voice-clone detection for calls, built for Indian languages. Team SONIX, SIH 2026. |

---

## On-screen captions

| Time | Caption |
|---|---|
| 0:15 | Real voice → GREEN |
| 0:35 | AI-cloned voice → RED |
| 0:55 | Flags, never blocks |
| 1:05 | Live risk timeline |
| 1:25 | Multiple models, one service |

---

## Rules (do not break)

- **No accuracy, EER or latency numbers** anywhere in the video.
- **Do not say it works on normal phone calls.** It works on web calls (Meet) and contact-centre/VoIP calls.
- **Do not say it detects "any" AI voice.** Say it *flags likely AI-cloned voices*.
- It **never blocks** a call. Always say it *warns* and a *person decides*.
- The audio clips shown must be ones the team has rights to use.

---

## What I need from you (Claude)

1. A **shot list** for recording, with exact screen setups and click steps.
2. **Title card** and **end card** text and a simple layout (SONIX name, tagline, "Team SONIX · SIH 2026").
3. An **.srt subtitle file** for the voice-over, timed to the script above.
4. **Editing notes**: where to cut, zoom in on the band, and trim waiting time so the video stays at 2:00.
5. A **checklist** for the live take: server running, extension loaded, Meet noise cancellation off, clips rehearsed, wait about 4 s after each clip starts before talking about the result.
