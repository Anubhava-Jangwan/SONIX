# head_v3 runbook — from embeddings to a model in the live UI

**Hand this whole file to your AI assistant. It is a complete, ordered procedure.**
Written 3 Sept 2026. Every command runs from the repo root on Windows with the venv's python
(`.venv\Scripts\python.exe`, or plain `python` if the venv is activated).

---

## 0. Why we are doing this

Our detector was trained on English (ASVspoof 2019 LA) plus channel augmentation. Tested on
Indian speech it flagged **the majority of genuine Indian speakers as fake**.

A previous fix added IndicVoices (real Indian speech) to training and cut that to near zero —
but that model then **passed a cloned Amitabh Bachchan clip as genuine**. Cause: every Indian
sample it had seen was labelled real, so it learned *"Indian audio = real"* rather than learning
to detect synthesis.

`head_v3` fixes it properly by putting Indian languages on **both sides of the label**, from
**three different synthesis families**.

---

## 1. State — verified on disk

| Root | Bonafide | Spoof | What it is |
|---|---|---|---|
| `embeddings` | 2,580 | 22,800 | ASVspoof 2019 LA, clean |
| `embeddings_g711` | 2,580 | 22,800 | + telephone codec |
| `embeddings_rawboost` | 2,580 | 22,800 | + channel/impulsive noise |
| `embeddings_rirmusan_bonafide` | 2,580 | 0 | + room reverb & MUSAN noise |
| `embeddings_rirmusan_spoof` | 0 | 22,800 | ” |
| `embeddings_indicvoices_tr` | **37,032** | 0 | **genuine Indian speech** |
| `embeddings_mms_tts` | 0 | 800 | Indic TTS (Meta MMS), 5 languages |
| `embeddings_mms_tts_aug` | 0 | 800 | ” + channel augmentation |
| `embeddings_indicsynth` | 0 | 16,800 | Indic voice conversion (freevc24), 12 languages |
| `embeddings_indicsynth_aug` | 0 | 16,800 | ” + channel augmentation |
| **TRAINING TOTAL** | **47,352** | **126,400** | ratio 2.67:1 |

**Indic slice specifically: 37,032 real vs 35,200 fake — 1.05:1.** That balance is the whole
point. Before this round it was 37,032 real vs **zero** fake, and that ratio *was* the bug.

**Held out, never trained on:** `embeddings_indicvoices_ho` (5,050 rows, split by recording).

---

## 2. Hard rules — violating any of these invalidates the result

1. **Never train on:** `embeddings_indicvoices_ho`, ASVspoof 2021 DF, In-the-Wild, Akshat's
   labelled clips, or the Bachchan pair. They are how we prove the model works.
2. **Never overwrite an existing checkpoint.** Back up `outputs\models` before starting.
3. **EER, not accuracy.** The data is ~73 % spoof; accuracy would be flattering and meaningless.
4. **Never invent a number.** If a run did not produce it, it does not exist.
5. **Watch the `pos_weight` line** `train.py` prints. It should be near `n_neg/n_pos`. If it is
   wildly off, a root is mislabelled — stop and check.

---

## 3. Back up first

```
xcopy outputs\models %USERPROFILE%\sonix_models_backup_v3\ /E /I /Y
```

`outputs\models` has been wiped three times this week. Five seconds now saves a night later.

---

## 4. Train the production head

One command. The frozen front-end already did the expensive work, so this is minutes, not hours.

```
python src/train.py ^
  --emb-root outputs/embeddings ^
  --extra-emb-root outputs/embeddings_g711 ^
  --extra-emb-root outputs/embeddings_rawboost ^
  --extra-emb-root outputs/embeddings_rirmusan_bonafide ^
  --extra-emb-root outputs/embeddings_rirmusan_spoof ^
  --extra-emb-root outputs/embeddings_indicvoices_tr ^
  --extra-emb-root outputs/embeddings_mms_tts ^
  --extra-emb-root outputs/embeddings_mms_tts_aug ^
  --extra-emb-root outputs/embeddings_indicsynth ^
  --extra-emb-root outputs/embeddings_indicsynth_aug ^
  --out outputs/models/head_v3.pt
```

`--extra-emb-root` uses `action="append"`, so repeating it genuinely stacks the roots.

**What to check in the output:**
- Total vectors ≈ **173,752**
- Class counts ≈ 47,352 bonafide / 126,400 spoof
- `pos_weight` ≈ **0.37**
- A `BEST dev EER` line — **write it down**, it goes in `docs/PPT_MASTER.md` §8

Then confirm the file is a real checkpoint:
```
python inspect_ckpt.py outputs/models/head_v3.pt
```
Expect `parameters: 262,657` and a config line.

---

## 5. Train four ablations — this is the results slide

Each takes minutes and answers *"which data actually helped?"*. Without these we can only say
"we threw everything in".

```
:: A — no Indic spoof at all (reproduces the bug, this is the "before")
python src/train.py --emb-root outputs/embeddings ^
  --extra-emb-root outputs/embeddings_g711 ^
  --extra-emb-root outputs/embeddings_rawboost ^
  --extra-emb-root outputs/embeddings_rirmusan_bonafide ^
  --extra-emb-root outputs/embeddings_rirmusan_spoof ^
  --extra-emb-root outputs/embeddings_indicvoices_tr ^
  --out outputs/models/abl_a_no_indic_spoof.pt

:: B — + MMS-TTS only        (does one TTS system alone fix it?)
:: C — + IndicSynth only     (does voice conversion alone fix it?)
:: D — everything but the channel-augmented copies  (did augmentation matter?)
```

B, C and D are the same command as §4 with the relevant `--extra-emb-root` lines removed.

---

## 6. THE GATE — this decides everything

Run every head against the Bachchan pair and any Hindi real/clone clips.

```
python bench_clips.py --real <hindi real folder> --fake <hindi clone folder>
```

Four possible outcomes:

| Real Bachchan | Cloned Bachchan | Meaning | Action |
|---|---|---|---|
| real | **fake** | ✅ **Fixed.** Ship it. | Go to §8 |
| fake | fake | Still "Indian = fake" | Too little Indic bonafide, or spoofs too clean — check the augmented roots were included |
| real | real | Still "Indian = real" | Too little Indic spoof, or a stamping bug — re-run every `--check` |
| fake | real | Inverted — something is mislabelled | Stop. Verify every root's labels before anything else |

**Also read the SNR-correlation column** `bench_clips.py` prints. It is the correlation between a
genuine clip's SNR and its score. Near 0 means the old "clean background = fake" shortcut is dead.
Above +0.6 means it is still alive and no threshold will fix it.

---

## 7. Confirm nothing regressed

```
python src/eval.py --split train --emb-root outputs/embeddings_indicvoices_ho ^
    --model-ckpt outputs/models/head_v3.pt --out-scores outputs/scores_ho_v3
python flag_rate.py --split train --scores-dir outputs/scores_ho_v3
```

This is the **held-out** Indian speech, split by recording, never trained on. It measures genuine
Indian false alarms honestly.

⚠️ **Phrase the result precisely:** `flag_rate.py` counts one score per 4-second chunk with no
smoothing. So it is *"X % of genuine 4-second chunks"*, never *"X % of calls"* — the live path
applies a 5-window moving average and alarms less often.

Then, if the DF21 embeddings are available (Navya's machine), re-score cross-dataset EER against
`head_v3` and always quote it with **"(75 % key coverage)"**.

If DF21 got materially worse, say so and consider dropping a root. Do not quietly keep the number
that flatters us.

---

## 8. Wire it into the live UI

**Register the model.** Edit `realtime/models.py` and add to `REGISTRY`:

```python
    "v3": (
        "SONIX v3 (multilingual)",
        "outputs/models/head_v3.pt",
        "Clean + G.711 + RawBoost + RIR/MUSAN + IndicVoices + 35,200 Indic spoofs "
        "across three synthesis families. Indian languages on both sides of the label.",
    ),
```

Then set it as the default:
```python
DEFAULT_KEY = "v3"
```

**Verify the registry sees it:**
```
python -c "import sys; sys.path.insert(0,'.'); from realtime import models; print(models.DEFAULT_KEY); [print(' ', m['key'], m['exists']) for m in models.catalogue()]"
```
`v3` must show `exists=True`.

**Start the server** (no `--ckpt` — it picks the default itself):
```
python -m realtime.server --auto-approve --ws-port 8000
```
Wait for `Engine: front-end warm.`

**Start the dashboard:**
```
streamlit run realtime/live_ui.py
```

**Verify in the UI:**
- Sidebar lists **SONIX v3** and it is selected by default
- Upload a genuine clip → graph builds window by window → stays **GREEN**
- Upload a cloned clip → climbs to **RED**
- Switching models re-scores the same clip differently

**Standard demo settings:** model `v3` · Amber 0.10 · Red 0.90 · silence gate `Auto` ·
smoothing 5-window mean with 3-of-5 hysteresis (fixed, not a control).

---

## 9. Update the record

Put the new numbers in `docs/PPT_MASTER.md` §8, moving rows from ⏳ PENDING to 🔒 LOCKED:

- `[[MODEL_NAME]]` → `head_v3` / "SONIX v3"
- `[[EER_INDOMAIN]]` → the dev EER from §4
- `[[FA_INDIC]]` → the flag rate from §7, phrased per 4-second chunk
- `[[FA_REAL]]` → from `bench_clips.py`
- The ablation table from §5

Commit and push so Yukti's deck picks it up:
```
git add docs/PPT_MASTER.md realtime/models.py outputs/models/head_v3.pt
git commit -m "Train head_v3 on multilingual corpus; register as default model"
git push
```

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `pos_weight` far from 0.37 | A root is mislabelled — run `stamp_labels.py --check` on each |
| Dev EER ≈ 0 % | Leakage. Check the holdout root is not in the training list |
| Dev EER ≈ 40–50 % | Label bug. Some root has its classes inverted |
| CUDA out of memory during training | `--batch 128`. Training is small; this is rare |
| `no shards in ...` | Wrong path, or the copy from the SSD is incomplete |
| Model missing from the UI sidebar | `exists=False` — check the path in `models.py` matches the filename |
| Server starts in mock mode | No checkpoint found; check `outputs/models/head_v3.pt` exists |

---

## 11. If the gate fails and time runs out

**Do not demo on Indian audio.** The model genuinely works on the English material it was trained
and tested on. Demo that, and present the Indic gap as measured, diagnosed, and in progress with
the corpus already built.

A team that says *"we tested our own system on Indian speech, found it failed, built a
35,200-clip corpus to fix it, and here is the measurement"* is far stronger than one that gets
caught by a judge holding a Hindi clip.

**What we must not do is ship a model whose Indic numbers look good because it says "real" to
everything Indian.** We already know that failure mode by name.
