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
