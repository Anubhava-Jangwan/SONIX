# SONIX -- Results Table (single source of truth)

**SIH26104 - Team Sonix - Stage 2 PoC**
Owner: Yukti. Every number here is reproducible by re-running `src/metrics.py`.
If a number is not in this file, it is not established -- do not put it on a slide.

---

## A. Confirmed -- safe to present

**All rows below independently reproduced from Yugal's score arrays via
`metrics.py`, 1 Sept 2026.** Two implementations (his `eval.py` cross-check and
this one) agree to four decimal places.

| Result | Value | How to reproduce |
|---|---|---|
| ASVspoof-2019 LA **eval** EER, baseline `head.pt` | **1.4937%** @ threshold 0.095130 | `python src\metrics.py --split eval` |
| eval set size | 71,237 (7,355 bonafide / 63,882 spoof) | printed by the same command |
| ASVspoof-2019 LA dev EER, baseline `head.pt` | 0.1592% @ threshold 0.836843 | `python src\metrics.py --split dev` |
| dev set size | 24,844 (2,548 bonafide / 22,296 spoof) | printed by the same command |
| Augmented head `head_aug.pt`, dev EER | 0.24% (early-stopped, epoch 11) | Yugal's training log |
| Augmented training set | 50,760 vectors (25,380 clean + 25,380 G.711) | -- |
| Calibration temperature | T ~ 1.05 (grid value 1.049750, dev BCE loss 0.004075) | `python src\calibration.py --fit-split dev --apply-split eval` |
| Calibration effect on EER | **none** -- 1.4937% before and after | `python src\metrics.py --split eval --scores-file outputs\scores\eval_calibrated_scores.npy` |
| Calibrated EER threshold | 0.104725 (vs 0.095130 uncalibrated) | same command -- only the threshold moves |
| Demo path sanity check | real 0.0070 / fake 0.9820 | `python demo\score_file.py <clip>` |

### Padding fix -- measured before/after
Same genuine human recording (`chetan voice.ogg`, 66.1 s), baseline `head.pt`,
identical excerpt positions. Score 0-1, higher = model thinks it is fake.

| Clip length | Before (zero-padded) | After (repeat-padded) |
|---|---|---|
| 1 second | **0.8795** | **0.0329** |
| 2 seconds | 0.5061 | 0.0005 |
| 3 seconds | 0.2224 | 0.0096 |
| full 66 s | 0.1171 | 0.1171 *(no padding involved -- control)* |

Full-length run: median window score 0.0064; only 2% of its 125 windows scored above 0.90.
Plot: `python src\padding_plot.py` -> `outputs/plots/padding_fix.png`

---

## B. Not yet measured -- leave blank, do not estimate

| Result | Owner | Status |
|---|---|---|
| ASVspoof-2021 DF EER (baseline / augmented) | Navya | pending -- blocked on the DF CM key file |
| In-the-Wild EER (baseline / augmented) | Akshat | pending |
| Codec 2x2 (baseline/aug x clean/G.711) | Yugal | pending |
| Real-clip false-alarm rate, before vs after the padding fix | Yugal + Yukti | pending -- blocked on Akshat's labelled clips |
| Real-world amber / red operating point | Yukti | pending -- same blocker |

---

## C. Caveats that must appear on the slide

1. **The augmented model's 0.24% dev EER is NOT comparable to the baseline's
   0.16%.** The augmented dev set includes codec copies -- they are different
   test sets. Presenting them side by side without this note is the single
   easiest thing for a sharp judge to catch.
2. **Dev is never a headline number.** Dev shares attack types with training, so
   a near-zero EER there is expected, not evidence of quality. It is a sanity
   check only.
3. **Calibration did not improve accuracy, and was never meant to.** It makes the
   0-1 output a real probability so a threshold means the same thing across
   datasets. Our fitted T came out very close to 1, meaning the model was
   already well calibrated in-domain. That is a legitimate finding -- and it also
   means **calibration is not the fix for our real-world false alarms.**
4. **T is known only to +/-0.0125.** `calibration.py` grid-searches T over
   `linspace(0.05, 100.0, 4000)`, a step of ~0.025. Quoting "T = 1.0497" implies
   four-decimal precision the search cannot deliver. Say **T ~ 1.05**.
5. **Cross-dataset EER will look bad, and we report it as-is.** 10-30% on DF21
   and In-the-Wild is the normal, correct, publishable result for this setup.
   The honesty is a strength and the PPT already frames it as a known risk.
6. **The metric is EER, not accuracy.** The data is ~90% spoof, so accuracy
   would be flattering and meaningless.
