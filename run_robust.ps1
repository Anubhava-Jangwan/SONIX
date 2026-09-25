# run_robust.ps1 -- SONIX / SIH26104
#
# Fixes the checkpoint-name trap, trains the four-source robust head,
# scores it on ASVspoof-2021 DF, and prints the comparison.
#
# Run from the repo root with the venv active:
#     cd D:\SONIX
#     .\.venv\Scripts\Activate.ps1
#     .\run_robust.ps1
#
# Stops at the first failure rather than charging on with bad data.

$ErrorActionPreference = "Stop"

function Step($n, $msg) {
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor DarkGray
    Write-Host "  STEP $n  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor DarkGray
}

function Die($msg) {
    Write-Host ""
    Write-Host "FAILED: $msg" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "src\train.py")) {
    Die "run this from D:\SONIX (I can't see src\train.py from here)"
}

# ---------------------------------------------------------------------------
Step 1 "Repair the checkpoint names"
# head.pt / head_aug.pt are DIRECTORIES (a .pt is a zip; someone unzipped it).
# train.py, eval.py and the demo all DEFAULT to outputs/models/head.pt, so an
# empty folder there is a live demo-crash risk.

Push-Location outputs\models
foreach ($pair in @(@("head.pt", "head_rebuilt.pt"), @("head_aug.pt", "head_aug_rebuilt.pt"))) {
    $name, $src = $pair
    if (-not (Test-Path $src)) { Pop-Location; Die "$src is missing" }
    if (Test-Path $name -PathType Container) {
        Remove-Item -Recurse -Force $name
        Write-Host "  removed stale directory $name"
    }
    Copy-Item $src $name -Force
    $sz = (Get-Item $name).Length
    Write-Host "  $name  <- $src  ($sz bytes)" -ForegroundColor Green
}
Pop-Location

# ---------------------------------------------------------------------------
Step 2 "Verify every embedding root before training"
# The single most expensive mistake available tonight is training on a folder
# whose spoof clips are still labelled bonafide. Check, do not assume.

$roots = @(
    @{ path = "F:\embeddings";                            expect = "mixed" },
    @{ path = "F:\embeddings_g711";                       expect = "mixed" },
    @{ path = "F:\embeddings_rawboost";                   expect = "mixed" },
    @{ path = "outputs\embeddings_rirmusan_bonafide";     expect = "all0"  },
    @{ path = "outputs\embeddings_rirmusan_spoof";        expect = "all1"  }
)

foreach ($r in $roots) {
    $d = Join-Path $r.path "train"
    if (-not (Test-Path $d)) { Die "$d does not exist" }
    Write-Host ""
    Write-Host "  $($r.path)  (expect: $($r.expect))" -ForegroundColor Yellow
    python stamp_labels.py --emb-dir $d --check
    if ($LASTEXITCODE -ne 0) { Die "label check failed on $d" }
}

Write-Host ""
Write-Host "  Read the counts above before continuing." -ForegroundColor Yellow
Write-Host "  mixed -> roughly bonafide 2580 / spoof 22800" -ForegroundColor Yellow
Write-Host "  all0  -> bonafide 2580,  spoof 0" -ForegroundColor Yellow
Write-Host "  all1  -> bonafide 0,     spoof 22800" -ForegroundColor Yellow
Write-Host ""
$ans = Read-Host "  Do the counts look right? (y/n)"
if ($ans -ne "y") { Die "stopped by you -- fix the labels before training" }

# ---------------------------------------------------------------------------
Step 3 "Train the four-source robust head"
# clean + G.711 codec + RawBoost + RIR/MUSAN  ~= 101,500 vectors

python src\train.py `
    --emb-root F:\embeddings `
    --extra-emb-root F:\embeddings_g711 `
    --extra-emb-root F:\embeddings_rawboost `
    --extra-emb-root outputs\embeddings_rirmusan_bonafide `
    --extra-emb-root outputs\embeddings_rirmusan_spoof `
    --out outputs\models\head_robust_v2.pt
if ($LASTEXITCODE -ne 0) { Die "training failed" }

Write-Host ""
Write-Host "  SANITY CHECK on the dev EER printed above:" -ForegroundColor Yellow
Write-Host "    low fractions of a percent = working" -ForegroundColor Yellow
Write-Host "    ~40%  = label bug, STOP" -ForegroundColor Yellow
Write-Host "    ~0%   = data leak, STOP" -ForegroundColor Yellow

# ---------------------------------------------------------------------------
Step 4 "Score on ASVspoof-2021 DF"

python src\eval.py --split eval `
    --emb-root outputs\embeddings_df21 `
    --model-ckpt outputs\models\head_robust_v2.pt `
    --out-scores outputs\scores_df21_robust
if ($LASTEXITCODE -ne 0) { Die "DF21 scoring failed" }

python src\metrics.py --split eval `
    --scores-dir outputs\scores_df21_robust `
    --out-dir outputs\plots_df21\robust
if ($LASTEXITCODE -ne 0) { Die "metrics failed" }

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 68) -ForegroundColor DarkGray
Write-Host "  COMPARE against the measured numbers" -ForegroundColor Cyan
Write-Host ("=" * 68) -ForegroundColor DarkGray
Write-Host ""
Write-Host "    baseline  head.pt        DF21 EER = 9.4835 %"
Write-Host "    augmented head_aug.pt    DF21 EER = 8.5126 %"
Write-Host "    robust_v2 (4 sources)    DF21 EER = see above"
Write-Host ""
Write-Host "  All three on the same 400,435 trials, 75% coverage of the DF key." -ForegroundColor DarkGray
Write-Host "  Quote it as '... on 75% coverage', never as the bare DF21 EER." -ForegroundColor DarkGray
Write-Host ""
Write-Host "  If robust_v2 is WORSE than 8.5126%, report it as-is." -ForegroundColor DarkGray
Write-Host "  'We measured four augmentation strategies and two helped' is a" -ForegroundColor DarkGray
Write-Host "  stronger methodology story than one lucky number." -ForegroundColor DarkGray
Write-Host ""
