# SONIX — real-time voice-clone detection

**SIH26104 · AICTE Cyber Security Cell** — AI-powered detection of voice-cloning
impersonation attacks. Team SONIX.

SONIX listens to call audio and scores how likely the speaker is AI-generated,
showing a Green / Amber / Red risk band that refreshes about twice a second. It
never blocks a call — it flags for a human to verify.

**How it works.** A frozen wav2vec2 (XLS-R 300M) front-end turns each 4-second
audio window into one 1024-dim embedding (mean-pooled over time). A small trained
MLP head (~300k params) scores that embedding. Only the head is trained — that's
what makes single-GPU training and real-time inference possible. Metric is **EER**
(Equal Error Rate), not accuracy, so it's comparable to published baselines.

Because the front-end is frozen and shared, **swapping detection heads is nearly
free** — it loads a ~300k-param MLP and nothing else. That is why the demo can
change model mid-sentence without interrupting the graph, and why the live server
can re-point a call at a different head without dropping it.

---

## TL;DR — get something on screen

```bash
# from the repo root, with the venv active
./venv/Scripts/streamlit.exe run demo/app.py --server.port 8501
```

Open <http://localhost:8501>. That single page does uploads, live microphone
scoring, head comparison, and the Chrome-extension setup. Nothing else needs to
be running for the first three.

---

## The three surfaces

| Surface | Command | Port | What it is |
|---|---|---|---|
| **Demo dashboard** | `streamlit run demo/app.py` | 8501 | The main UI: analyze a file, live mic, compare heads, approve extension calls |
| **Detection server** | `python -m realtime.server …` | 8000 | WebSocket/HTTP scoring service. Only the Chrome extension needs it |
| **Chrome extension** | load unpacked from `extension/` | — | Scores a Google Meet call live, in a side panel |

> There is no longer a separate operator dashboard. `realtime/live_ui.py` was
> removed; everything it did — including approving pairing codes — now lives in
> the demo dashboard's **Extension** section.

---

## Setup

Python 3.11–3.13. Create an isolated environment, then install PyTorch **with
CUDA** (a plain `pip install torch` can give a CPU-only build):

```bash
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/Mac:  source venv/bin/activate

# torch + torchaudio from the CUDA index. Pick the build for your driver:
#   Python <=3.12 & older driver:  cu121
#   Python 3.13 / newer driver:    cu124 or cu126   (cu121 has NO 3.13 wheels)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Confirm the GPU is visible (must print `True` and your GPU name):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**On Windows, use the venv's interpreter explicitly.** A bare `python` is usually
the system install and will not have these dependencies:

```bash
./venv/Scripts/python.exe   -m realtime.server ...
./venv/Scripts/streamlit.exe run demo/app.py
```

---

## 1 — The demo dashboard (port 8501)

```bash
./venv/Scripts/streamlit.exe run demo/app.py --server.port 8501
```

One page, scrolled, with an anchor nav: **Overview · Live mic · Analyze ·
Compare · Method · Extension**.

### Live mic

Continuous capture at 16 kHz, scored window by window, in this process — no
second service and no second port. Press **Start listening**.

- **Input device picker.** Choose a microphone to score your own voice, or a
  loopback device (*Stereo Mix*, *What U Hear*) to score whoever is on a call.
- **Siri bubble.** Three rings deformed by the live audio envelope, plus a dBFS
  reading against the silence gate. If it moves, audio is arriving. If it is
  grey and still, nothing is reaching the mic. This is the first thing to look
  at when no scores appear.
- **Band thresholds.** Amber and Red sliders. They re-read scores already
  computed — nothing is re-scored, so the graph repaints instantly.
- **Change head mid-stream.** The graph keeps going; each point keeps the head
  that actually scored it.

The first **Start** takes 30–60 s while the 300M front-end loads. The panel says
so. Afterwards, head swaps are instant.

> Capture needs `sounddevice` (in `requirements.txt`). It opens the device at
> its **native** rate and resamples to 16 kHz — asking a 44.1 kHz device for
> 16 kHz fails outright on some Windows audio backends.

### Analyze / Compare

Upload a `wav/flac/mp3/ogg`, or pick a bundled clip from `demo_clips/`. Compare
runs the same clip through several heads; each head costs one more full pass of
the frozen front-end, so the default set is three.

---

## 2 — The detection server (port 8000)

Only needed by the Chrome extension.

```bash
./venv/Scripts/python.exe -m realtime.server \
    --ckpt outputs/models/head_v3.pt \
    --ws-port 8000 --mode webrtc \
    --vad-energy 0.003
```

Check it: <http://localhost:8000/api/status> → `"scoring_available": true`.

### Flags that matter

| Flag | Why |
|---|---|
| `--ckpt PATH` | Which head. Without it the server runs but reports `scoring_available: false` and no number is ever shown |
| `--ws-port 8000` | WebSocket + HTTP port. The extension defaults to this |
| `--mode webrtc` | Browser/extension capture. Other modes: `voip`, `upload` |
| `--vad-energy 0.003` | **Silence-gate floor.** See below — this is the single most common reason nothing gets scored |
| `--auto-approve` | Skip the consent gate. Demo only; it bypasses a real safety feature |

### The silence gate

Windows below an energy floor never reach the model, because near-silence looks
like nothing the model was trained on and it tends to call that "fake".

The default floor (`0.01` RMS ≈ −40 dBFS) is tuned for clean close-mic speech.
**Google Meet tab audio is already noise-suppressed and AGC'd and lands far
below that**, so with the default almost every window is rejected and the graph
sits empty while audio is plainly flowing. `--vad-energy 0.003` (≈ −50 dBFS) is
a reasonable starting point.

The trade-off is real: a lower floor means more readings but more false alarms
during quiet passages. If red bands appear while nobody is speaking, raise it.

Diagnose it with telemetry, which reports both the measurement and the setting:

```bash
curl -s http://localhost:8000/api/telemetry
# calls.<id>.vad → windows_seen, windows_passed, pass_rate,
#                  threshold_energy, and last.{rms,peak}
```

`windows_seen` climbing while `windows_passed` stays frozen means audio is
arriving and being rejected as silence — lower the floor.

---

## 3 — The Chrome extension

Scores a Google Meet call live in a half-transparent side panel.

**It captures the TAB's audio — the other participants, not your microphone.**
That is the right way round: you are detecting whether the person calling *you*
is synthetic. There is exactly one capture path and it is bound to
`chromeMediaSource: "tab"`; no microphone permission is requested.

### Install

1. Start the detection server (section 2) and the dashboard (section 1)
2. Open `chrome://extensions` — Chrome or Edge 116+
3. Turn on **Developer mode** (top right)
4. **Load unpacked** → select the `extension/` folder
5. Pin the SONIX icon

### Use

1. Join a Meet call, then **reload the tab**. The content script only injects at
   page load, so an already-open tab will not show the panel until you refresh.
2. **Tell the other participants they are being monitored.** The panel is on
   your screen only; Meet gives them no indication.
3. SONIX icon → **Start monitoring**
4. A six-digit pairing code appears in the popup and the panel. Approve it at
   <http://localhost:8501> → **Extension** → step 2. **Codes expire after 120 s**
   — if yours has, press **New code** in the panel.
5. The panel shows the verdict (**Authentic / Uncertain / Cloned**), a live
   graph, a head picker and Amber/Red sliders.

Changing the head does **not** restart the call — the `call_id`, the consent
grant and the graph all survive the swap.

### After editing any extension file

`chrome://extensions` → **↻** on the SONIX card, **then `Ctrl+R` on the Meet
tab**. Both. The reload arrow alone does not re-inject `content.js`, and this is
the most common reason a change appears not to have worked.

---

## Troubleshooting

**Nothing is scored; `windows scored` stays 0.**
Check the call state first. `CONSENT_PENDING` means the code was never approved
and the server is discarding every sample by design. If the state is `scoring`
but nothing moves, it is the silence gate — see section 2.

**The panel shows an old layout / missing buttons.**
The extension was not fully reloaded. See "After editing any extension file".

**"Cannot capture a tab with an active stream."**
A previous capture stream is still held. Press **Force-stop capture stream** in
the popup. It survives the popup closing, the extension reloading and the tab
navigating, so this is the only clean way out short of restarting Chrome.

**The popup connects to nothing and the server log is silent.**
Check the **Server** field in the popup. The stored value overrides the default
permanently, so one bad paste persists. It must be `http://localhost:8000` — not
the dashboard's `:8501`. The dot next to the field turns red and says so; press
**reset**.

**The live mic never scores.**
Watch the Siri bubble. Grey and still = no audio reaching the device (wrong
input device?). Moving while the graph stays flat = audio is under the silence
gate.

**PortAudio: "Error querying device -1" / "Invalid device".**
Some Windows machines expose inputs only through WDM-KS, which does no sample-
rate conversion, and report an unusable default device. `demo/live.py` resolves
a real input and opens it at its native rate. Use the **Input device** picker if
the automatic choice is wrong.

---

## Training and evaluation

### The data

| Dataset | Role | Where it goes |
|---|---|---|
| ASVspoof 2019 LA | training + in-domain eval | `data/asvspoof19_la/` |
| In-the-Wild | unseen-attack eval only | `data/in_the_wild/` |

`data/asvspoof19_la/` must **directly** contain `ASVspoof2019_LA_train`,
`ASVspoof2019_LA_dev`, `ASVspoof2019_LA_eval` and `ASVspoof2019_LA_cm_protocols`.
The official `LA.zip` unzips into a doubled `LA/LA/…` — flatten it rather than
passing a deeper path.

Verify completeness **before** any long run. A partial pendrive copy or an
interrupted download surfaces later as mass `! missing source` errors, not as an
upfront failure:

```bash
find data/asvspoof19_la/ASVspoof2019_LA_eval/flac -name "*.flac" | wc -l   # 71933
```

> The datasets are not in this repo — LA alone is ~7.12 GB. `outputs/` is
> generated locally and also stays out of git.

### Run order

```bash
# 1. Gate — never skip. Must print PASS.
python src/verify_protocol.py

# 2. Cache embeddings (the slow step, done once). Smoke-test 50 files first:
python src/extract_embeddings.py --split train --batch 8 --limit 50   # expect (50, 1024)
python src/extract_embeddings.py --split train --batch 8
python src/extract_embeddings.py --split dev   --batch 8
python src/extract_embeddings.py --split eval  --batch 8              # 4 GB cards: --batch 4

# 3. Train the head (minutes — embeddings are cached)
python src/train.py

# 4. THE number: EER on the official eval split
python src/eval.py --split eval

# 5. Cross-dataset: In-the-Wild
python src/extract_embeddings.py --split itw --batch 8 --itw-root data/in_the_wild
python src/eval.py --split itw
```

`src/extract_embeddings.py` is **resumable** — it saves a shard every 100 files and
skips finished shards, so if it crashes just re-run the same command. Never
re-extract from scratch. On CUDA out-of-memory, lower `--batch`.

### Sanity check on the EER

- **low single digits** → working
- **~40%** → a bug, almost always the protocol parse or a flipped label → re-run the gate
- **exactly 0%** → eval leaked into training
- **In-the-Wild / codec-degraded much worse than clean eval** → **expected**.
  That generalization gap is the project's headline finding, not a bug to hide.

Always report EER as "X% on \<split name\>", never as a bare number.

---

## The scoring interface

```python
from score_file import score_file
scores = score_file("call.wav")   # one prob per 4 s window @ 0.5 s hop; higher = faker
```

Smoothing, hysteresis and the Green/Amber/Red mapping live in the UI layer
(`demo/risk.py`, `demo/ui.py`).

---

## Repository layout

```
SIH_sonix/
├─ demo/                     # the dashboard on 8501
│  ├─ app.py                 #   the single page
│  ├─ live.py                #   continuous mic capture + scoring threads
│  ├─ core.py                #   scoring state, model registry access
│  ├─ ui.py                  #   design system: CSS, charts, Siri orb, verdict
│  ├─ score_file.py          #   the scoring interface (frozen front-end + heads)
│  └─ test_live.py           #   hardware-free checks for the live path
├─ extension/                # Chrome MV3 extension
│  ├─ background.js          #   capture lifecycle, state, message routing
│  ├─ offscreen.js           #   the only place audio is touched
│  ├─ content.js             #   the in-call side panel
│  └─ popup.html/.css/.js    #   toolbar control surface
├─ realtime/                 # the detection service on 8000
│  ├─ server.py              #   WebSocket + HTTP entrypoint
│  ├─ engine.py              #   model owner, batches across calls
│  ├─ session.py             #   per-call state machine + audit
│  ├─ models.py              #   the head registry (9 heads)
│  ├─ ringbuffer.py          #   4 s windows @ 0.5 s hop
│  ├─ vad.py                 #   the silence gate
│  └─ resample.py            #   anything → 16 kHz
├─ src/                      # training / evaluation
├─ data/                     # NOT in git — datasets
└─ outputs/                  # NOT in git — embeddings, models, scores, call audits
```

---

## Privacy and consent

Audio leaves the browser only to `localhost` and is never sent anywhere else.
The server holds it in a four-second ring buffer, scores it, and writes an audit
record of the call's **events — not the audio** — to `outputs/calls/`.

The consent gate is real: a session discards every chunk until its pairing code
is approved, and the refusal is recorded. That is a design commitment, not a UI
convention, and `--auto-approve` exists only for demos you control.

None of which removes your obligation to tell people in the call that you are
monitoring it. Recording law varies by jurisdiction and by whether all parties
consent. For the hackathon demo, use your own team.
