#!/usr/bin/env bash
# Train / evaluate / export the Korean streaming zipformer KWS (icefall gigaspeech/KWS recipe).
#   bash run.sh --stage 0 --stop-stage 0            # train (defaults = the released tiny2 run)
#   bash run.sh --stage 1 --stop-stage 1            # CER + KWS report on the 5 test sets (CPU)
#   bash run.sh --stage 2 --stop-stage 2            # export pretrained.pt + streaming ONNX (fp32/int8)
set -eou pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; . ./env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
stage=0; stop_stage=100; epoch=${EPOCH:-12}; avg=${AVG:-1}; exp=${EXP:-exp/pt5k-tiny2}; start_epoch=1
arch=${ARCH_PRESET:-tiny2}; train_cuts=${TRAIN_CUTS:-pt5k}; LANGD=${LANG_DIR:-data/lang_syl_k2000}; FBANK=${FBANK_DIR:-data/fbank-k}
lr_epochs=${LR_EPOCHS:-3.5}; max_duration=${MAX_DURATION:-1600}
while [ $# -gt 0 ]; do case "$1" in
  --stage) stage=$2; shift 2;; --stop-stage) stop_stage=$2; shift 2;; --epoch) epoch=$2; shift 2;; --avg) avg=$2; shift 2;;
  --exp) exp=$2; shift 2;; --arch) arch=$2; shift 2;; --start-epoch) start_epoch=$2; shift 2;; --train-cuts) train_cuts=$2; shift 2;;
  --lr-epochs) lr_epochs=$2; shift 2;; --max-duration) max_duration=$2; shift 2;;
  *) echo "unknown $1"; exit 1;; esac; done
log() { echo -e "$(date '+%H:%M:%S') [run] $*"; }
run() { [ $stage -le $1 ] && [ $stop_stage -ge $1 ]; }

# tiny  = 3.3M zipformer of icefall egs/gigaspeech/KWS (with a 2,000-token vocabulary: 5.1M)
# tiny2 = released model: decoder/joiner 320→192 and two layers per stack (6.6M incl. training-only parts)
# small = two layers per stack, wider encoder (15.7M)
case "$arch" in
  tiny)  ARCH=(--decoder-dim 320 --joiner-dim 320 --num-encoder-layers 1,1,1,1,1,1 --feedforward-dim 192,192,192,192,192,192
               --encoder-dim 128,128,128,128,128,128 --encoder-unmasked-dim 128,128,128,128,128,128 --causal 1);;
  tiny2) ARCH=(--decoder-dim 192 --joiner-dim 192 --num-encoder-layers 2,2,2,2,2,2 --feedforward-dim 192,192,192,192,192,192
               --encoder-dim 128,128,128,128,128,128 --encoder-unmasked-dim 128,128,128,128,128,128 --causal 1);;
  small) ARCH=(--decoder-dim 320 --joiner-dim 320 --num-encoder-layers 2,2,2,2,2,2 --feedforward-dim 384,384,512,512,384,384
               --encoder-dim 192,192,256,256,192,192 --encoder-unmasked-dim 192,192,192,192,192,192 --causal 1);;
  *) echo "unknown arch $arch"; exit 1;;
esac

if run 0; then
  log "stage 0: train $exp ($arch, $train_cuts, $epoch epochs)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" zipformer/train.py \
      --world-size 1 --exp-dir "$exp" "${ARCH[@]}" \
      --num-epochs "$epoch" --start-epoch "$start_epoch" --lr-epochs "$lr_epochs" --use-fp16 1 \
      --bpe-model $LANGD/bpe.model --manifest-dir "$FBANK" --train-cuts "$train_cuts" \
      --enable-musan 1 --max-duration "$max_duration" --num-workers ${NUM_WORKERS:-6}
fi

if run 1; then
  log "stage 1: report (epoch $epoch avg $avg)"
  SETS="${SETS:-kspon_eval_clean kspon_eval_other fleurs_test aihub_cmd_test aihub_noisy_test}" \
      bash local/kws_bench.sh "$exp" "$arch" "$epoch" "$avg"
fi

if run 2; then
  log "stage 2: export"
  "$PY" zipformer/export.py --epoch "$epoch" --avg "$avg" --exp-dir "$exp" --tokens $LANGD/tokens.txt "${ARCH[@]}" --chunk-size 16 --left-context-frames 64
  "$PY" zipformer/export-onnx-streaming.py --epoch "$epoch" --avg "$avg" --exp-dir "$exp" --tokens $LANGD/tokens.txt "${ARCH[@]}" --chunk-size 16 --left-context-frames 128
  ls -la "$exp"/*.onnx
fi
log "done"
