#!/usr/bin/env bash
# After pretraining ends: pick the best averaged model on dev, evaluate on the held-out test sets,
# export streaming ONNX (int8) and write a summary. CPU-only so it never competes for the GPU.
#   local/finalize.sh exp/pt5k-tiny2 tiny2 12
set -eou pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd "$KWS_TRAIN"
exp=$1; arch=$2; last=$3
export CUDA_VISIBLE_DEVICES="" LANG_DIR=data/lang_syl_k2000 FBANK_DIR=data/fbank-k
case "$arch" in
  small) ARCH="--decoder-dim 320 --joiner-dim 320 --num-encoder-layers 2,2,2,2,2,2 --feedforward-dim 384,384,512,512,384,384 --encoder-dim 192,192,256,256,192,192 --encoder-unmasked-dim 192,192,192,192,192,192 --causal 1";;
  tiny2) ARCH="--decoder-dim 192 --joiner-dim 192 --num-encoder-layers 2,2,2,2,2,2 --feedforward-dim 192,192,192,192,192,192 --encoder-dim 128,128,128,128,128,128 --encoder-unmasked-dim 128,128,128,128,128,128 --causal 1";;
  tiny)  ARCH="--decoder-dim 320 --joiner-dim 320 --num-encoder-layers 1,1,1,1,1,1 --feedforward-dim 192,192,192,192,192,192 --encoder-dim 128,128,128,128,128,128 --encoder-unmasked-dim 128,128,128,128,128,128 --causal 1";;
esac
S=$exp/summary.txt; : > $S
cer() { # epoch avg testset
  "$PY" zipformer/decode-asr.py --epoch $1 --avg $2 --use-averaged-model 1 --exp-dir $exp --bpe-model $LANG_DIR/bpe.model --manifest-dir $FBANK_DIR $ARCH \
      --chunk-size 16 --left-context-frames 64 --test-set $3 --decoding-method greedy_search --max-duration 300 > /dev/null 2>&1
  "$PY" local/cer.py "$(ls -t $exp/greedy_search/recogs-$3-*epoch-$1-avg-$2-* | head -1)" | cut -d' ' -f2
}
# 1. choose avg on dev (kspon_dev) — test sets stay untouched until the choice is made
best=""; bestc=999
for avg in 1 3 5 8; do
  c=$(cer $last $avg kspon_dev); echo "dev kspon_dev epoch $last avg $avg: CER $c" | tee -a $S
  v=${c%\%}; if awk "BEGIN{exit !($v < $bestc)}"; then bestc=$v; best=$avg; fi
done
echo "chosen: epoch $last avg $best (dev CER $bestc%)" | tee -a $S
# 2. full report (CER + KWS FRR, FA/h, FRR@FA/h) on the held-out test sets
OUT_SUFFIX=-final SETS="${SETS:-kspon_eval_clean kspon_eval_other fleurs_test aihub_cmd_test aihub_noisy_test}" bash local/kws_bench.sh $exp $arch $last $best > $exp/bench-final.log 2>&1
cat $exp/bench-e$last-a$best-final/report.txt | tee -a $S
# 4. export: pretrained.pt (for fine-tuning) + streaming ONNX int8
"$PY" zipformer/export.py --epoch $last --avg $best --exp-dir $exp --tokens $LANG_DIR/tokens.txt $ARCH --chunk-size 16 --left-context-frames 64 > $exp/export.log 2>&1
"$PY" zipformer/export-onnx-streaming.py --epoch $last --avg $best --exp-dir $exp --tokens $LANG_DIR/tokens.txt $ARCH --chunk-size 16 --left-context-frames 128 >> $exp/export.log 2>&1
ls -la $exp/pretrained.pt $exp/*.onnx | awk '{print $5, $9}' | tee -a $S
echo "FINALIZE DONE $(date +%H:%M)" | tee -a $S
