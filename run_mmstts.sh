#!/usr/bin/env bash
# Yugal's slice of the Indic fix: generate synthetic Indian speech locally with
# MMS-TTS, extract, stamp as spoof. No download -- safe to run alongside the
# ASVspoof 5 transfer, since that is network-bound and this is GPU-bound.
set -u
cd "$(dirname "$0")"
LOG="mmstts_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "=== $(date '+%H:%M:%S')  $* ==="; }
die() { echo; echo "!!! STOPPED: $*"; exit 1; }

source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || die "venv"
pip install -q -U transformers scipy 2>&1 | tail -1

say "generating (5 languages x 160 clips)"
for L in hin mar ben tam tel; do
  python make_indic_spoof.py --lang "$L" --n 160 --out data/indic_spoof \
    || echo ">>> $L failed, continuing"
done
N=$(ls data/indic_spoof 2>/dev/null | wc -l); echo "clips: $N"
[ "$N" -gt 200 ] || die "only $N clips generated"

say "extracting"
python src/extract_embeddings.py --split train --audio-dir data/indic_spoof \
    --out outputs/embeddings_mms_tts --batch 8 || die "extraction"

say "stamping as SPOOF"
python stamp_labels.py --emb-dir outputs/embeddings_mms_tts/train --label 1 || die "stamp"
echo; echo "--- VERIFY (must be all spoof = 1) ---"
python stamp_labels.py --emb-dir outputs/embeddings_mms_tts/train --check

say "DONE"; echo "Log: $LOG"
