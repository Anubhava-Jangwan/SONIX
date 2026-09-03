#!/usr/bin/env bash
# Unattended: ASVspoof 5 TRAIN split -> labelled embedding roots.
#
#   bash run_asvspoof5.sh
#
# Downloads only flac_T_* (train) and the protocols -- about 40 GB of the 142 GB
# repo. Never touches flac_E_* (eval): that is a test set and training on it
# would destroy the one number that proves we generalise.
#
# Safe to re-run: the download resumes and each stage skips work already done.

set -u
cd "$(dirname "$0")"
LOG="asvspoof5_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

say()  { echo; echo "=================================================="; \
         echo "$(date '+%H:%M:%S')  $*"; echo "=================================================="; }
die()  { echo; echo "!!! STOPPED: $*"; echo "log: $LOG"; exit 1; }

say "1/6  environment"
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null \
  || die "could not activate .venv"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  || die "torch import failed"
pip install -q -U "huggingface_hub[cli]" 2>&1 | tail -1

# prep_asvspoof5.py lives on navya/indic. Without it there is no way to turn the
# protocol into labels, and a flat folder would be stamped ALL bonafide.
[ -f prep_asvspoof5.py ] || die "prep_asvspoof5.py missing -- run: git merge origin/navya/indic"

say "2/6  download (train split + protocols only, ~40 GB, resumes)"
hf download jungjee/asvspoof5 --repo-type dataset --local-dir data/asvspoof5 \
   --include "flac_T_*.tar" "ASVspoof5_protocols.tar" "README.txt" "LICENSE.txt" \
   || die "download failed -- if it says gated, accept the terms on the dataset page then 'hf auth login'"

say "3/6  unpack"
mkdir -p data/asvspoof5/flac_T
for f in data/asvspoof5/flac_T_*.tar; do
  [ -e "$f" ] || die "no flac_T_*.tar found -- download did not complete"
  echo "  $f"; tar -xf "$f" -C data/asvspoof5/flac_T || die "untar $f"
done
tar -xf data/asvspoof5/ASVspoof5_protocols.tar -C data/asvspoof5 || die "untar protocols"
echo "audio files: $(find data/asvspoof5/flac_T -name '*.flac' | wc -l)"

say "4/6  locating the TRAIN protocol"
PROTO=$(find data/asvspoof5 -name "*.tsv" | grep -i train | head -1)
[ -n "$PROTO" ] || { echo "tsv files found:"; find data/asvspoof5 -name "*.tsv"; \
                     die "no *train*.tsv -- tell Claude which one to use"; }
echo "using: $PROTO"

say "5/6  splitting into bonafide/ and spoof/ (hardlinks, no extra disk)"
AUD=$(find data/asvspoof5/flac_T -name '*.flac' | head -1 | xargs -r dirname)
[ -n "$AUD" ] || die "no .flac files under data/asvspoof5/flac_T"
echo "audio dir: $AUD"
python prep_asvspoof5.py --protocol "$PROTO" --audio-dir "$AUD" \
    --out-dir data/asvspoof5_split/train || die "prep_asvspoof5"

for c in bonafide spoof; do
  n=$(ls "data/asvspoof5_split/train/$c" 2>/dev/null | wc -l)
  echo "  $c: $n files"
  [ "$n" -gt 100 ] || die "$c has only $n files -- protocol/audio mismatch, do NOT train on this"
done

say "6/6  extract + stamp"
python src/extract_embeddings.py --split train \
    --audio-dir data/asvspoof5_split/train/bonafide \
    --out outputs/embeddings_as5_bonafide --batch 8 || die "extract bonafide"
python src/extract_embeddings.py --split train \
    --audio-dir data/asvspoof5_split/train/spoof \
    --out outputs/embeddings_as5_spoof --batch 8 || die "extract spoof"

python stamp_labels.py --emb-dir outputs/embeddings_as5_bonafide/train --label 0 || die "stamp bonafide"
python stamp_labels.py --emb-dir outputs/embeddings_as5_spoof/train    --label 1 || die "stamp spoof"

echo; echo "--- VERIFY bonafide (must be all 0) ---"
python stamp_labels.py --emb-dir outputs/embeddings_as5_bonafide/train --check
echo; echo "--- VERIFY spoof (must be all 1) ---"
python stamp_labels.py --emb-dir outputs/embeddings_as5_spoof/train --check

say "DONE"
echo "Read the two VERIFY blocks above before training on these roots."
echo "Log: $LOG"
