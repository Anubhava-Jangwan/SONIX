# SONIX Prototype Video Script (2 min)

---

## PART 1: GOOGLE MEET (0:00 – 1:02)

### 0:00 · Opening
**Screen:** Meet call running, SONIX extension visible

> This is SONIX, live on a Google Meet call.
> It listens to the other person and flags if their voice is AI-cloned.

---

### 0:08 · Start monitoring
**Do:** Click SONIX icon → **Start monitoring** → approve

> We turn on monitoring.
> From here, SONIX checks the call every half second.

---

### 0:15 · Real voice
**Other person:** Unmute → play **REAL** clip
**Wait:** about 4 seconds · band stays **GREEN**

> First, a real human voice.
> The band stays green, so nothing suspicious.

---

### 0:35 · AI-cloned voice
**Other person:** Play **FAKE** clip
**Wait:** about 4 seconds · band goes **AMBER → RED**

> Now an AI-cloned voice.
> Within a few seconds, SONIX turns red: likely synthetic.

---

### 0:55 · Key point
**Screen:** Hold on the red overlay

> It doesn't cut the call.
> It warns you, and you decide.

---

## PART 2: STREAMLIT UI (1:02 – 2:00)

### 1:02 · Switch screens
**Do:** Switch to the Streamlit console

> Everything you just saw runs through the SONIX service.
> Here's the operator console.

---

### 1:05 · Call timeline
**Screen:** This call's timeline

> This is the same call as a timeline:
> green while the real voice spoke, red when the clone started.

---

### 1:25 · Compare models
**Do:** Upload the fake clip → run several heads side by side

> We can also test any recording with several detection models side by side.
> When a new cloning tool appears, we add a new model without touching the service.

---

### 1:45 · Close
**Screen:** SONIX logo + team name

> SONIX: real-time AI voice-clone detection for calls,
> built for Indian languages.
> Team SONIX, SIH 2026.

---

## BEFORE RECORDING

- [ ] Server running, console open, extension loaded
- [ ] Rehearse once with the same 2 clips (real = green, fake = red)
- [ ] Other person: **turn off Meet noise cancellation**, play clips at normal volume
- [ ] After each clip starts, **wait about 4 s** before speaking about the result
