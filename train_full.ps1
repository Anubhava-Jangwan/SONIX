# train_full.ps1 -- SONIX / SIH26104
#
# Trains head_full.pt on ALL FIVE training sources:
#     clean + G.711 + RawBoost + RIR/MUSAN + IndicVoices
# then scores it on ASVspoof-2021 DF and on ASVspoof-2019 LA eval,
# and prints the full comparison.
#
#     cd D:\SONIX
#     .\.venv\Scripts\Activate.ps1
#     .\train_full.ps1
#
# All data now lives on D: -- no external drive needed.

$ErrorActionPreference = "Stop"

function Step($n, $msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host "  STEP $n  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray
}
function Die($msg) {
    Write-Host ""; Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1
}

if (-not (Test-Path "src\train.py")) { Die "run this from D:\SONIX" }

# ---------------------------------------------------------------------------
Step 0 "Check the training roots are present"

if (-not (Test-Path "D:\embeddings\train")) {
    Write-Host ""
    Write-Host "  D:\embeddings\train is not reachable." -ForegroundColor Red
    Write-Host "  Expected the copied clean embeddings there." -ForegroundColor Yellow
    exit 1
}
Write-Host "  all roots reachable on D:" -ForegroundColor Green

# ---------------------------------------------------------------------------
Step 1 "Verify labels on every training root"
# The one mistake that silently ruins a run is a spoof folder still carrying
# the --audio-dir placeholder label of 0. Check, do not assume.

$roots = @(
    @{ p = "D:\embeddings";                        want = "mixed  ~2580 bona / ~22800 spoof" },
    @{ p = "D:\embeddings_g711";                   want = "mixed  ~2580 bona / ~22800 spoof" },
    @{ p = "D:\embeddings_rawboost";               want = "mixed  ~2580 bona / ~22800 spoof" },
    @{ p = "outputs\embeddings_rirmusan_bonafide"; want = "ALL 0  2580 bona / 0 spoof"        },
    @{ p = "outputs\embeddings_rirmusan_spoof";    want = "ALL 1  0 bona / 22800 spoof"       },
    @{ p = "outputs\embeddings_indicvoices";       want = "ALL 0  ~21040 bona / 0 spoof"      }
)

foreach ($r in $roots) {
    $d = Join-Path $r.p "train"
    if (-not (Test-Path $d)) { Die "$d does not exist" }
    Write-Host ""
    Write-Host "  $($r.p)" -ForegroundColor Yellow
    Write-Host "    expect: $($r.want)" -ForegroundColor DarkGray
    python stamp_labels.py --emb-dir $d --check
    if ($LASTEXITCODE -ne 0) { Die "label check failed on $d" }
}

Write-Host ""
$ans = Read-Host "  Do all six match what they should be? (y/n)"
if ($ans -ne "y") { Die "stopped by you -- fix the labels first" }

# ---------------------------------------------------------------------------
Step 2 "Train head_full.pt on all five sources"
# IndicVoices is bonafide-only, which shifts the class ratio from ~9:1 toward
# ~3:1. train.py recomputes pos_weight from the combined set, so this is
# handled -- and a less lopsided ratio is generally healthier.

python src\train.py `
    --emb-root D:\embeddings `
    --extra-emb-root D:\embeddings_g711 `
    --extra-emb-root D:\embeddings_rawboost `
    --extra-emb-root outputs\embeddings_rirmusan_bonafide `
    --extra-emb-root outputs\embeddings_rirmusan_spoof `
    --extra-emb-root outputs\embeddings_indicvoices `
    --out outputs\models\head_full.pt
if ($LASTEXITCODE -ne 0) { Die "training failed" }

Write-Host ""
Write-Host "  SANITY CHECK the dev EER printed above:" -ForegroundColor Yellow
Write-Host "    low fractions of a percent = working" -ForegroundColor Yellow
Write-Host "    ~40%  = label bug, STOP" -ForegroundColor Yellow
Write-Host "    ~0%   = data leak, STOP" -ForegroundColor Yellow
Write-Host "  (dev EER is NOT comparable across models -- each one is" -ForegroundColor DarkGray
Write-Host "   validated on a different, harder dev set. DF21 is the" -ForegroundColor DarkGray
Write-Host "   comparable number.)" -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
Step 3 "Score on ASVspoof-2021 DF (cross-dataset -- the headline)"

python src\eval.py --split eval `
    --emb-root outputs\embeddings_df21 `
    --model-ckpt outputs\models\head_full.pt `
    --out-scores outputs\scores_df21_full
if ($LASTEXITCODE -ne 0) { Die "DF21 scoring failed" }

python src\metrics.py --split eval `
    --scores-dir outputs\scores_df21_full `
    --out-dir outputs\plots_df21\full
if ($LASTEXITCODE -ne 0) { Die "DF21 metrics failed" }

# ---------------------------------------------------------------------------
Step 4 "Score on ASVspoof-2019 LA eval (in-domain -- did we regress?)"

python src\eval.py --split eval `
    --emb-root D:\embeddings `
    --model-ckpt outputs\models\head_full.pt `
    --out-scores outputs\scores_la_full
if ($LASTEXITCODE -ne 0) { Die "LA eval scoring failed" }

python src\metrics.py --split eval `
    --scores-dir outputs\scores_la_full `
    --out-dir outputs\plots\la_full
if ($LASTEXITCODE -ne 0) { Die "LA metrics failed" }

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor DarkGray
Write-Host "  COMPARE" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor DarkGray
Write-Host ""
Write-Host "  DF21 EER (400,435 trials, 75% coverage, identical labels):"
Write-Host "    baseline    clean                              9.4835 %"
Write-Host "    head_aug    + G.711                            8.5126 %"
Write-Host "    robust_v2   + RawBoost + RIR/MUSAN             5.4956 %"
Write-Host "    head_indic  + IndicVoices                      (run it -- see below)"
Write-Host "    head_full   ALL FIVE                           see STEP 3"
Write-Host ""
Write-Host "  LA eval EER (in-domain, 71,237 trials):"
Write-Host "    baseline                                       1.4937 %"
Write-Host "    head_indic                                     2.0257 %"
Write-Host "    head_full                                      see STEP 4"
Write-Host ""
Write-Host "  A small in-domain rise is the expected price of robustness." -ForegroundColor DarkGray
Write-Host "  Report both columns -- the trade is the story, not a flaw." -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Still worth running (head_indic has never been scored on DF21):" -ForegroundColor Yellow
Write-Host "    python src\eval.py --split eval --emb-root outputs\embeddings_df21 ``"
Write-Host "        --model-ckpt outputs\models\head_indic.pt ``"
Write-Host "        --out-scores outputs\scores_df21_indic"
Write-Host "    python src\metrics.py --split eval --scores-dir outputs\scores_df21_indic ``"
Write-Host "        --out-dir outputs\plots_df21\indic"
Write-Host ""
