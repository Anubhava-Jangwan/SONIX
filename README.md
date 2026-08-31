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

---

## Repository layout

```
Sonix/
├─ extract_embeddings.py     # cache frozen wav2vec2 embeddings to disk
│                            #   (self-contained: the one file you copy to another machine)
├─ src/
│  ├─ verify_protocol.py     # gate: checks the dataset labels parse correctly
│  ├─ train.py               # train the MLP head on cached embeddings
│  ├─ eval.py                # EER on the eval split + In-the-Wild cross-dataset test
│  ├─ make_codec.py          # G.711 codec-degraded copy of a split, for robustness runs
│  └─ score_file.py          # score_file(wav)->list[float], the demo/UI interface
├─ requirements.txt
├─ data/                     # NOT in git — put the datasets here (see below)
│  ├─ asvspoof19_la/         #   ASVspoof 2019 LA (train/dev/eval)
│  └─ in_the_wild/           #   In-the-Wild (eval only)
└─ outputs/                  # NOT in git — created by the scripts
   ├─ embeddings/            #   cached shards (train/ dev/ eval/ itw/)
   ├─ models/                #   head.pt checkpoint
   └─ scores/                #   labels/scores .npy for the metric code
```

> **The datasets are not in this repo.** They're far too large for GitHub (LA is
> ~7 GB). Get them separately (pendrive or shared drive) and place them under
> `data/`. `outputs/` is generated locally and also stays out of git.

## The data

| Dataset | Role | Where it goes |
|---|---|---|
| ASVspoof 2019 LA | training + in-domain eval | `data/asvspoof19_la/` |
| In-the-Wild | unseen-attack eval only | `data/in_the_wild/` |

`data/asvspoof19_la/` must **directly** contain the `ASVspoof2019_LA_train`,
`ASVspoof2019_LA_dev`, `ASVspoof2019_LA_eval`, and `ASVspoof2019_LA_cm_protocols`
folders. (The official `LA.zip` unzips into a doubled `LA/LA/…` — flatten it.)

Canonical counts (the gate checks these): **train 25380 · dev 24844 · eval 71237**.

---

## Setup

Python 3.11–3.13. Create an isolated environment, then install PyTorch **with
CUDA** (a plain `pip install torch` can give a CPU-only build):

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac:  source .venv/bin/activate

# torch + torchaudio from the CUDA index. Pick the build for your driver:
#   Python <=3.12 & older driver:  cu121
#   Python 3.13 / newer driver:    cu124   (cu121 has no 3.13 wheels)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Confirm the GPU is visible (must print `True` and your GPU name):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Run order

Every script defaults to the folders above, so from the repo root:

```bash
# 1. Gate — never skip this. Must print PASS.
python src/verify_protocol.py

# 2. Cache embeddings (the slow step; done once). Smoke-test 50 files first:
python extract_embeddings.py --split train --batch 8 --limit 50   # expect shape (50, 1024)
python extract_embeddings.py --split train --batch 8
python extract_embeddings.py --split dev   --batch 8
python extract_embeddings.py --split eval  --batch 8               # 4 GB cards: --batch 4

# 3. Train the head (minutes, because embeddings are cached)
python src/train.py

# 4. THE number: EER on the official eval split
python src/eval.py --split eval

# 5. Cross-dataset: In-the-Wild (expected to be much worse — that's the headline)
python extract_embeddings.py --split itw --batch 8 --itw-root data/in_the_wild
python src/eval.py --split itw
```

`extract_embeddings.py` is **resumable** — it saves a shard every 100 files and
skips finished shards on restart, so if it crashes just re-run the same command.
It also never dies on one bad audio file. On CUDA out-of-memory, lower `--batch`.

### Sanity check on the EER
- **low single digits** → working
- **~40%** → a bug, almost always the protocol parse or a flipped label → re-run the gate
- **exactly 0%** → eval leaked into training

## For the demo (Suryansh)

```python
from score_file import score_file
scores = score_file("call.wav")   # one prob per 4s window @ 0.5s hop; higher = more likely fake
```

Smoothing, hysteresis, and the Green/Amber/Red mapping live in the UI layer.

---

## Live Detection (Real-time VoIP / WebRTC)

**NEW:** Real-time scoring on live phone calls. Capture audio from Asterisk PBX (SIP) or browser WebRTC, embed with frozen wav2vec2, batch-score with MLP head, display timeline in Streamlit dashboard.

### Quick Start

```bash
# Terminal 1: Start server (mock mode, no head.pt needed)
python -m realtime.server --mock --port 5000 --ws-port 8000

# Terminal 2: Open live dashboard
streamlit run realtime/live_ui.py

# Terminal 3 (optional): Upload a WAV file for scoring
curl -X POST http://localhost:8000/api/score-file \
  -F "file=@sample_call.wav"
```

### Architecture

```
Audio Capture
├── Asterisk/SIP (VoIP) → AudioSocket protocol
├── Browser (WebRTC) → WebSocket frames
└── File Upload → HTTP POST

        ↓

realtime/ringbuffer.py (4-second windows, 0.5s overlap)
        ↓
realtime/vad.py (energy + zero-crossing, skip silence)
        ↓
wav2vec2 (frozen, pre-trained) → 1024-dim embedding
        ↓
realtime/engine.py (single model owner, batches 1–8 calls)
        ↓
head.pt (trained MLP, ~300k params) → P(AI)
        ↓
realtime/live_ui.py (Streamlit dashboard, WebSocket feed)
```

### Files

**Core Realtime Module:**
- `realtime/session.py` — Per-call state machine (CONNECTING → CONSENT_PENDING → LISTENING → SCORING → ENDED)
- `realtime/engine.py` — Model owner, batches windows across concurrent calls, broadcasts scores
- `realtime/server.py` — Entrypoint: AudioSocket TCP server + WebSocket broadcast + HTTP file upload
- `realtime/live_ui.py` — Streamlit dashboard: pairing code, consent status, score timeline, file upload

**Existing Infrastructure (Pre-written):**
- `realtime/audiosocket.py` — AudioSocket TCP framing (Asterisk → app)
- `realtime/ringbuffer.py` — 4-second buffer, overlapping windows
- `realtime/resample.py` — 8kHz → 16kHz conversion
- `realtime/vad.py` — Voice activity detection
- `realtime/source.py` — Audio source abstraction
- `realtime/consent.py` — Consent state machine + audit logging
- `realtime/pairing.py` — 6-digit device pairing codes
- `realtime/checkpoint.py` — head.pt validation
- `realtime/mock.py` — Deterministic mock scorer (for testing without head.pt)

### Server Flags

```bash
python -m realtime.server \
  --port 5000          # AudioSocket TCP port (Asterisk)
  --ws-port 8000       # WebSocket + HTTP port (UI + uploads)
  --mock               # Use mock scorer (no head.pt needed)
  --ckpt models/head.pt # Path to trained head.pt (optional, for real scoring)
  --mode voip|webrtc|upload  # Capture source (default: voip)
  --max-calls 4        # Max concurrent calls
  --host 0.0.0.0       # Bind address
  --output-dir outputs/calls  # Where to save call records
```

### Live UI Features

- **Pairing Code** — 6-digit code, 120s expiry, HMAC-validated
- **Consent Status** — Shows pairing approval state
- **Score Timeline** — P(AI) per window, updates every 0.5s
- **File Upload** — Post-call scoring of .wav files
- **Call History** — Recent calls + metadata
- **Server Status** — Active calls, engine stats, diagnostics

### WebSocket API

**Messages sent by server:**

```json
{
  "type": "pairing_request",
  "call_id": "sip_001",
  "pairing_code": "123456",
  "expires_in": 120,
  "caller": "+1-508-799-1234"
}

{
  "type": "scores",
  "timestamp": "2026-09-01T12:34:56.789Z",
  "data": {
    "sip_001": {"window_idx": 10, "score": 0.32}
  }
}
```

**Messages accepted by server:**

```json
{"type": "approve_pairing", "call_id": "sip_001"}
{"type": "end_call", "call_id": "sip_001"}
{"type": "ping"}
```

### HTTP Endpoints

- `GET /api/status` — Server status + engine stats
- `POST /api/score-file` — Upload WAV, returns scores
- `GET /ws` — WebSocket endpoint (connect for live scores)

### Testing

```bash
# Run pytest
pytest realtime/tests/ -v

# Test individual modules
pytest realtime/tests/test_session.py -v
pytest realtime/tests/test_engine.py -v
```

### Known Limitations

- **No real scoring yet** — head.pt not trained (pending Yugal's ML work)
- **Consent-only** — Scoring blocked until user approves pairing code
- **Local network** — Asterisk runs on-premise, WebRTC on localhost
- **Mock mode default** — Use `--ckpt` to enable real scoring once head.pt exists

### Roadmap

- [ ] Week 1 (Sept 1): Core files + tests + Streamlit UI ✓
- [ ] Week 2: Asterisk docker-compose setup guide
- [ ] Week 2: WebRTC ingest module (browser mic capture)
- [ ] Week 3: Benchmarking (p50/p95 latency per concurrent call)
- [ ] Week 4 (Yugal): fp16 precision fix + head.pt training
- [ ] Week 4: Swap mock → real scoring
- [ ] Future: Google Meet / Teams integration

---

## Contributing

**Your Lane (Capture → Embed → Score → Display):**
- realtime/ module (session, engine, server, live_ui)
- WebRTC ingest, file upload, Streamlit dashboard
- Asterisk setup + benchmarking

**Yugal's Lane (ML Training):**
- head.pt training
- Model architecture, loss, optimization
- fp16 precision fixes
- Real scoring integration

**Anubhav's Lane (Dataset):**
- Embedding extraction (fp16 fix pending)
- Codec robustness (G.711, Opus degradation)
- EER validation + metrics

**Suryansh's Lane (Demo UI):**
- Streamlit refinement
- score_file integration with live_ui.py

---

## Status

- ✓ Architecture ready (11/13+ core files written)
- ✓ Session state machine implemented
- ✓ Scoring engine batching implemented
- ✓ Server entrypoint + WebSocket broadcast implemented
- ✓ Streamlit dashboard implemented
- ✓ Mock scorer (testing without head.pt)
- ⏳ Asterisk docker-compose setup guide (Week 2)
- ⏳ WebRTC ingest module (Week 2)
- ⏳ head.pt training (Yugal, Week 4)
- ⏳ Real scoring (after head.pt)

**Current:** Mock mode works end-to-end. Ready for testing and integration.

