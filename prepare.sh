#!/usr/bin/env bash
# Data preparation → data/fbank-k/kws_cuts_{pt5k,dev,<test sets>}.jsonl.gz
#   . ./env.sh && bash prepare.sh [--stage N] [--stop-stage M]
# Stages are idempotent. Downloads go to $RAW (see README "데이터 받기").
set -eou pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; . ./env.sh
stage=0; stop_stage=100
while [ $# -gt 0 ]; do case "$1" in --stage) stage=$2; shift 2;; --stop-stage) stop_stage=$2; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
log() { echo -e "$(date '+%H:%M:%S') [prepare] $*"; }
run() { [ $stage -le $1 ] && [ $stop_stage -ge $1 ]; }
M=$DATA/manifests; F=$DATA/fbank; FK=$DATA/fbank-k; LK=$DATA/lang_syl_k2000
mkdir -p "$M" "$F" "$FK" "$LK"

if run 0; then
  log "stage 0: check / extract downloads in $RAW"
  for f in zeroth_korean.tar.gz musan.tar.gz; do [ -s "$RAW/$f" ] || { log "missing $RAW/$f"; exit 1; }; done
  [ -d "$RAW/zeroth_korean" ] || { mkdir -p "$RAW/zeroth_korean"; tar xzf "$RAW/zeroth_korean.tar.gz" -C "$RAW/zeroth_korean"; }
  [ -d "$RAW/musan" ] || tar xzf "$RAW/musan.tar.gz" -C "$RAW"
  [ -f "$RAW/fleurs/data/ko_kr/train.tsv" ] || { log "missing FLEURS ko_kr under $RAW/fleurs"; exit 1; }
fi

if run 1; then
  log "stage 1: manifests — Zeroth-Korean, FLEURS ko_kr, MUSAN"
  ZROOT=$(find "$RAW/zeroth_korean" -maxdepth 3 -name AUDIO_INFO -printf '%h\n' | head -1)
  "$PY" local/prepare_zeroth.py --root "$ZROOT" --out "$M"
  "$PY" local/prepare_fleurs.py --root "$RAW/fleurs/data/ko_kr" --out "$M"
  [ -f "$M/musan_recordings_music.jsonl.gz" ] || "$VENV/bin/lhotse" prepare musan "$RAW/musan" "$M"
fi

if run 2; then
  log "stage 2: KsponSpeech manifests (AI Hub 123, extracted under $RAW/ksponspeech; PCM → FLAC next to it)"
  KROOT=$(find "$RAW/ksponspeech" -maxdepth 3 -name train.trn -printf '%h\n' | head -1)
  [ -n "$KROOT" ] || { log "KsponSpeech train.trn not found under $RAW/ksponspeech"; exit 1; }
  "$PY" local/prepare_kspon.py --corpus "$KROOT" --out "$M" --nj 16
fi

if run 3; then
  log "stage 3: fbank cut sets (Zeroth/FLEURS train speed-perturbed 0.9/1.1; Kspon not)"
  fb() { "$PY" local/compute_fbank.py --name "$1" --recordings "$M/$2_recordings_$3.jsonl.gz" \
             --supervisions "$M/$2_supervisions_$3.jsonl.gz" --out "$F" --perturb-speed "$4" ${5:+--max-dur $5}; }
  fb zeroth_train zeroth train 1
  fb fleurs_train fleurs train 1
  fb fleurs_dev   fleurs dev   0
  fb fleurs_test  fleurs test  0
  for p in train dev eval_clean eval_other; do fb kspon_$p kspon $p 0 25; done
  "$PY" local/compute_fbank_musan.py --manifests "$M" --out "$F"
fi

if run 4; then
  log "stage 4: syllable BPE-2000 (shipped in data/lang_syl_k2000; retrained only if missing)"
  if [ ! -f "$LK/bpe.model" ]; then
    "$PY" - "$M" "$LK/transcript.txt" <<'PYEOF'
import gzip, json, sys
sys.path.insert(0, "local"); from text_norm import compose
m, out = sys.argv[1], open(sys.argv[2], "w", encoding="utf-8")
for f in ["kspon_supervisions_train", "zeroth_supervisions_train", "fleurs_supervisions_train"]:
    for line in gzip.open(f"{m}/{f}.jsonl.gz", "rt", encoding="utf-8"):
        t = compose(json.loads(line)["text"]).strip()
        if t: out.write(t + "\n")
PYEOF
    "$PY" local/train_bpe_model.py --lang-dir "$LK" --transcript "$LK/transcript.txt" --vocab-size 2000 --char-coverage 0.9995
  fi
fi

if run 5; then
  log "stage 5: syllable-text cut sets in $FK (features shared with $F) + pretrain_mux / dev"
  "$PY" local/make_syllable_cuts.py --src "$F" --dst "$FK"
  "$PY" local/combine_cuts.py --fbank "$FK" --mux --extra pretrain_mux=kspon_train,zeroth_train,fleurs_train
  "$PY" local/combine_cuts.py --fbank "$FK" --extra dev=kspon_dev,fleurs_dev
fi

if run 6; then
  log "stage 6: AI Hub download + 16 kHz Opus conversion (approval per dataset; key in \$AIHUB_API_KEY or ~/.config/ko-kws/aihub.key)"
  "$PY" local/aihub_fetch.py --out "$RAW/aihub" 485 109 96 71405
fi

if run 7; then
  log "stage 7: AI Hub manifests + per-sentence cap 20 for the scripted command corpora"
  for k in 485 109 96 71405; do "$PY" local/prepare_aihub.py $k --root "$RAW/aihub" --out "$M"; done
  "$PY" local/cap_per_text.py "$M/aihub96" train 20
  "$PY" local/cap_per_text.py "$M/aihub71405" train 20
fi

if run 8; then
  log "stage 8: test sets aihub_cmd_test / aihub_noisy_test + training set kws_cuts_pt5k (test sentences removed)"
  "$PY" local/build_pt5k.py
fi

if run 9; then
  log "stage 9: icefall-style KWS test sets (shipped in data/kws-test; rebuilt only if missing)"
  for s in kspon_eval_clean kspon_eval_other fleurs_test aihub_cmd_test aihub_noisy_test; do
    [ -f "$DATA/kws-test/$s/keywords.txt" ] || "$PY" local/make_kws_testsets.py --fbank "$FK" --out "$DATA/kws-test" --sets $s
  done
fi
log "done"
