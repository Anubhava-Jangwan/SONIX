#!/usr/bin/env bash
# Unattended: back up models, then build THREE Indic spoof sources -- three
# different synthesis families, which is what actually buys generalisation.
#
#   1. MLAAD Indic   -- text-to-speech, up to 140 models
#   2. IndicSynth    -- voice conversion (freevc24), 12 languages
#   3. MMS-TTS       -- a third TTS system, generated locally, no download
#
# (Indic-CodecFake would be the fourth family but is not released yet -- the
#  paper's dataset link is still a placeholder.)
#
#   bash run_overnight.sh
#
# Safe to re-run: downloads resume, and each stage is skipped if its output
# already exists. Nothing here touches a test set or an existing model.

set -u
cd "$(dirname "$0")"
LOG="overnight_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

say()  { echo; echo "=================================================="; \
         echo "$(date '+%H:%M:%S')  $*"; echo "=================================================="; }
warn() { echo; echo ">>> SKIPPED: $*  (continuing -- the other sources still run)"; }

# A stage that fails must NOT kill the run: two good datasets beat zero.
STAGE_FAILED=""
note_fail() { STAGE_FAILED="$STAGE_FAILED\n  - $*"; warn "$*"; }

say "0/7  backing up outputs/models"
BK="$HOME/sonix_models_backup_$(date +%Y%m%d_%H%M)"
mkdir -p "$BK" && cp -v outputs/models/*.pt "$BK"/ && echo "backed up to $BK"

say "1/7  environment"
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || {
  echo "FATAL: could not activate .venv"; exit 1; }
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || {
  echo "FATAL: torch import failed"; exit 1; }
pip install -q -U huggingface_hub datasets soundfile scipy 2>&1 | tail -2

# ---------------------------------------------------------------- 1. MLAAD
say "2/7  MLAAD Indic -- download + flatten (TTS family)"
if [ -d data/mlaad_indic ] && [ "$(ls data/mlaad_indic 2>/dev/null | wc -l)" -gt 500 ]; then
  echo "already present, skipping download"
else
  python prep_mlaad.py --list && \
  python prep_mlaad.py --langs hi,bn,ta,mr,kn,ml,ur --out data/mlaad_indic --per-model 150 \
    || note_fail "MLAAD download"
fi

# ----------------------------------------------------------- 2. IndicSynth
say "3/7  IndicSynth -- streaming slice (voice-conversion family)"
if [ -d data/indicsynth ] && [ "$(ls data/indicsynth 2>/dev/null | wc -l)" -gt 500 ]; then
  echo "already present, skipping"
else
  python prep_indicsynth.py --list && \
  python prep_indicsynth.py --per-lang 1400 --out data/indicsynth \
    || note_fail "IndicSynth download"
fi

# -------------------------------------------------------------- 3. MMS-TTS
say "4/7  MMS-TTS -- generating locally (third TTS system, no download)"
if [ -d data/indic_spoof ] && [ "$(ls data/indic_spoof 2>/dev/null | wc -l)" -gt 300 ]; then
  echo "already present, skipping"
else
  for L in hin mar ben tam tel; do
    python make_indic_spoof.py --lang "$L" --n 160 --out data/indic_spoof \
      || note_fail "MMS-TTS $L"
  done
fi

# ---------------------------------------------------------------- extract
extract_and_stamp () {   # $1 = audio dir, $2 = embeddings root, $3 = human name
  local SRC="$1" DST="$2" NAME="$3"
  local N; N=$(ls "$SRC" 2>/dev/null | wc -l)
  if [ "$N" -lt 50 ]; then note_fail "$NAME: only $N clips, not extracting"; return; fi
  echo "$NAME: $N clips -> $DST"
  python src/extract_embeddings.py --split train --audio-dir "$SRC" \
      --out "$DST" --batch 8 || { note_fail "$NAME extraction"; return; }
  python stamp_labels.py --emb-dir "$DST/train" --label 1 || { note_fail "$NAME stamp"; return; }
  echo "--- $NAME VERIFY ---"
  python stamp_labels.py --emb-dir "$DST/train" --check || note_fail "$NAME check"
}

say "5/7  extracting MLAAD  (GPU)"
extract_and_stamp data/mlaad_indic  outputs/embeddings_mlaad_indic "MLAAD"

say "6/7  extracting IndicSynth  (GPU)"
extract_and_stamp data/indicsynth   outputs/embeddings_indicsynth  "IndicSynth"

say "7/7  extracting MMS-TTS  (GPU)"
extract_and_stamp data/indic_spoof  outputs/embeddings_mms_tts     "MMS-TTS"

say "DONE"
echo "Clip counts:"
for d in data/mlaad_indic data/indicsynth data/indic_spoof; do
  printf "  %-24s %s\n" "$d" "$(ls $d 2>/dev/null | wc -l)"
done
echo
echo "READ THIS: each VERIFY block above must say every row is spoof (1)."
echo "Any bonafide rows = do NOT train on that root, tell Claude."
if [ -n "$STAGE_FAILED" ]; then
  echo; echo "Stages that did not complete:"; echo -e "$STAGE_FAILED"
  echo "The rest finished -- re-run this script to retry just the missing ones."
fi
echo; echo "Log: $LOG"
