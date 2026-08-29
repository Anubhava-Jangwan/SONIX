# 00 - PROJECT CONTEXT (read this first)

## What the project is

Team SONIX - Smart India Hackathon 2026 - Problem statement SIH26104
"AI-powered real-time detection and prevention of voice cloning impersonation attacks".
Filed by AICTE's Cyber Security Cell. Team ID 590017752.

**The attack.** An attacker takes a few seconds of a CEO's voice from a public video. A
zero-shot voice cloning tool reproduces that voice with no training on that specific person.
They call the finance team with an urgent payment request using real project names. The
voice is recognised, the transfer is approved, the money is gone.

**Why nothing currently stops it.** Caller ID can be spoofed. Recognising the voice *is* the
attack - the more certain the victim is, the faster they comply. Fraud systems watch
transactions, not audio, so they fire only after the money has moved.

**What SONIX does.** Listens to live call audio, scores how likely the speaker is AI-generated,
and shows a Green / Amber / Red risk band that updates twice per second. Amber means
"call back on a number you already have before authorising." It never blocks a call - it flags
for human verification.

**What makes detection possible.** Neural speech synthesis ends in a vocoder that
reconstructs a waveform from a compressed description. Discarded detail is invented back,
wrongly and consistently - mostly in very high frequencies and phase. Humans cannot hear
these artifacts; a model trained on thousands of real and fake samples can measure them.

## Where we are

- Stage 1 (PPT round, 25-27 Aug): CLEARED.
- Stage 2 (PoC round): 1-2 September, physical, UPES campus. Top 45 teams advance.
- Judged on: solution depth, PoC/prototype, technical feasibility, overall impact.

## The pipeline

```
audio in -> 16 kHz mono, silence stripped
         -> sliding 4-second window, 0.5-second hop
         -> wav2vec2 (FROZEN - we do not retrain it)
         -> mean-pool to one 1024-dim vector per window
         -> small MLP classifier head (this is the only thing we train)
         -> score
         -> smoothing + hysteresis
         -> Green / Amber / Red band
```

**Why the front-end is frozen:** ASVspoof 2019 LA has ~25,000 training utterances against a
~300M-parameter model. Fine-tuning overfits fast and we do not have the GPU budget. We
train ~300k parameters instead of 300M - this is what makes single-GPU training and
real-time inference possible.

## The data

| Dataset | Size | Role |
|---------|------|------|
| ASVspoof 2019 LA | 7.12 GB | Training + in-domain evaluation |
| In-the-Wild | 8.16 GB | Unseen-attack evaluation only |
| ASVspoof 2021 DF | 34.5 GB | CUT - no time |

ASVspoof 2019 LA structure after extracting `LA.zip`:

```
LA/
  ASVspoof2019_LA_train/flac/   <- 25,380 files
  ASVspoof2019_LA_dev/flac/     <- 24,844 files
  ASVspoof2019_LA_eval/flac/    <- 71,237 files
  ASVspoof2019_LA_cm_protocols/
    ASVspoof2019.LA.cm.train.trn.txt
    ASVspoof2019.LA.cm.dev.trl.txt
    ASVspoof2019.LA.cm.eval.trl.txt
```

Protocol files are space-separated, five columns:
`speaker_id  filename  -  attack_id  label`
`label` is `bonafide` or `spoof`. Audio file is `filename + ".flac"`.

**Critical structural fact:** train and dev use attacks A01-A06. Eval uses A07-A19 -
deliberately different generation algorithms. That non-overlap is why generalisation is
testable at all.

Expected protocol counts:
- train: 25,380 lines - 2,580 bonafide / 22,800 spoof - attacks A01-A06
- dev: 24,844 lines
- eval: 71,237 lines - 7,355 bonafide / 63,882 spoof - attacks A07-A19

If bonafide and spoof counts are swapped, the labels are inverted and EER will be 100 minus
the real value.

## Our metric

**EER - Equal Error Rate.** Set the decision threshold so the rate of wrongly flagging genuine
speech equals the rate of missing fakes. That shared percentage is the EER. Lower is better.
Reported instead of accuracy because accuracy is misleading on imbalanced spoof data and
every published paper reports EER - so our number is directly comparable.

**Expected values.** In-domain (ASVspoof eval) should land in the low single digits. On
In-the-Wild it will be dramatically worse - the original paper reports up to a 1000% increase.
That drop is expected and is our headline finding, not a bug. It is called cross-dataset
generalisation and it is the acknowledged open problem of this research field.

## The team

| Person | Machine | Owns | Branch |
|--------|---------|------|--------|
| Yugal | RTX 3050 - 6 GB | All ML code, training, evaluation | `yugal/ml-core` |
| Anubhav | GTX 1650 - 4 GB | Eval-split extraction, opening pitch | `anubhav/eval-extraction` |
| Suryansh | no GPU | Demo UI, windowing, integration | `suryansh/demo-ui` |
| Navya | RTX 4060 - 8 GB | In-the-Wild extraction | `navya/itw-extraction` |
| Akshat | RTX 4060 - 8 GB | Demo audio generation, UI polish | `akshat/demo-audio` |
| Yukti | no GPU (CPU) | Evaluation code, EER metric, plots, jury prep | `yukti/evaluation` |

## Deliberately CUT - do not attempt

- Codec augmentation - needs a full re-extraction, another overnight run
- Indic evaluation set - a day of sourcing and generation, nobody free
- AASIST head / OC-Softmax loss - a broken upgrade the night before is worse than a working baseline
- ASVspoof 2021 DF - 34 GB, evaluation-only, zero value in three days
- Live microphone capture - file playback through the same pipeline looks identical on stage
- React, FastAPI, Docker, any API layer - Streamlit is the demo

On the slide these become "designed, not built" on a roadmap, which reads as scope discipline.

## Rules that apply to everyone

1. Never invent a number. One fabricated figure discounts everything else we said.
2. Cache everything expensive. Extracting embeddings is the slow part. Once cached, retraining is minutes.
3. Mock before you integrate. Anyone building UI works on fake scores from hour one.
4. If a script fails, send the full traceback. Do not paraphrase the error.
5. Ask before you improvise. If your task file does not cover something, message Yugal.
