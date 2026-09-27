#!/usr/bin/env bash
# Standard report for one checkpoint: CER on the 3 held-out sets + FRR, FA/h, FRR@FA/h.
#   local/kws_bench.sh <exp> <arch> <epoch> <avg> [score]
# CPU only. Writes <exp>/bench-e<epoch>-a<avg>/{report.txt,metrics.json,hits-*.jsonl}
set -eou pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd "$KWS_TRAIN"
exp=$1; arch=$2; ep=$3; avg=$4; score=${5:-1.0}
export CUDA_VISIBLE_DEVICES=""
LANGD=${LANG_DIR:-data/lang_syl_k2000}; FB=${FBANK_DIR:-data/fbank-k}; 
case "$arch" in
  small) ARCH="--decoder-dim 320 --joiner-dim 320 --num-encoder-layers 2,2,2,2,2,2 --feedforward-dim 384,384,512,512,384,384 --encoder-dim 192,192,256,256,192,192 --encoder-unmasked-dim 192,192,192,192,192,192 --causal 1";;
  tiny2) ARCH="--decoder-dim 192 --joiner-dim 192 --num-encoder-layers 2,2,2,2,2,2 --feedforward-dim 192,192,192,192,192,192 --encoder-dim 128,128,128,128,128,128 --encoder-unmasked-dim 128,128,128,128,128,128 --causal 1";;
  tiny)  ARCH="--decoder-dim 320 --joiner-dim 320 --num-encoder-layers 1,1,1,1,1,1 --feedforward-dim 192,192,192,192,192,192 --encoder-dim 128,128,128,128,128,128 --encoder-unmasked-dim 128,128,128,128,128,128 --causal 1";;
esac
AV="--use-averaged-model 1"   # icefall running-average model (clearly better than the raw epoch weights)
out=$exp/bench-e$ep-a$avg${OUT_SUFFIX:-}; mkdir -p $out; rm -f $out/hits-*.jsonl
SETS="${SETS:-kspon_eval_clean kspon_eval_other fleurs_test}"
: > $out/report.txt
echo "model: $exp epoch $ep avg $avg | streaming chunk 16 left 64 | keyword score $score" | tee -a $out/report.txt
echo "== CER (greedy)" | tee -a $out/report.txt
for ts in $([ "${SKIP_CER:-0}" = 1 ] || echo $SETS); do
  "$PY" zipformer/decode-asr.py --epoch $ep --avg $avg $AV --exp-dir $exp --bpe-model $LANGD/bpe.model --manifest-dir $FB $ARCH \
      --chunk-size 16 --left-context-frames 64 --test-set $ts --decoding-method greedy_search --max-duration 300 > /dev/null 2>&1
  r=$(ls -t $exp/greedy_search/recogs-$ts-*epoch-$ep-avg-$avg-* | head -1)
  echo "  $ts: $("$PY" local/cer.py $r)" | tee -a $out/report.txt
  [ "$ts" = fleurs_test ] && "$PY" local/fleurs_len.py $r | tee -a $out/report.txt
done
echo "== KWS, icefall-style per test set (keywords = top-20 nouns of that set; decoded at τ=0.01, swept offline)" | tee -a $out/report.txt
for ts in $SETS; do
  T=data/kws-test/$ts
  "$PY" zipformer/decode.py --epoch $ep --avg $avg $AV --exp-dir $exp --bpe-model $LANGD/bpe.model --manifest-dir $FB $ARCH \
      --chunk-size 16 --left-context-frames 64 --test-set $ts --keywords-file $T/keywords.txt --keywords-score $score --keywords-threshold 0.01 \
      --hits-file $out/hits-$ts.jsonl --max-duration 300 > $out/decode-$ts.log 2>&1
  "$PY" local/kws_metrics.py --meta $T/meta.json --hits $out/hits-$ts.jsonl --op 0.35 --at 1.0 --at 0.1 --json $out/metrics-$ts.json | tee -a $out/report.txt
done
