#!/usr/bin/env bash
# Keep exp/pt5k-tiny2 training alive: the main train.py process grows ~9 GB/h RSS (native memory,
# pre-existing issue) and gets OOM-killed by its MemoryMax. When the unit dies before epoch-12.pt
# exists, resume from the last finished epoch; also keep the per-epoch bench watcher running.
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd "$KWS_TRAIN"
EXP=exp/pt5k-tiny2; TOTAL=12
while [ ! -f $EXP/epoch-$TOTAL.pt ]; do
  if ! systemctl --user is-active -q kws-pt5k; then
    n=$(ls $EXP/epoch-*.pt | sed 's/.*epoch-//; s/.pt//' | sort -n | tail -1)
    echo "$(date +%m-%d_%H:%M) training unit down → resume from epoch $((n + 1))"
    [ -f exp-train-pt5k.log ] && mv exp-train-pt5k.log $EXP/train-until-e$n-$(date +%H%M).log
    local/job.sh pt5k 52G "cd $PWD && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LANG_DIR=data/lang_syl_k2000 FBANK_DIR=data/fbank-k NUM_WORKERS=6 bash run.sh --stage 0 --stop-stage 0 --arch tiny2 --train-cuts pt5k --epoch $TOTAL --start-epoch $((n + 1)) --lr-epochs 3.5 --max-duration 1600 --exp $EXP > exp-train-pt5k.log 2>&1"
    sleep 30
  fi
  if ! systemctl --user is-active -q kws-watch-pt5k; then
    local/job.sh watch-pt5k 12G "cd $PWD && SETS='kspon_eval_clean kspon_eval_other fleurs_test aihub_cmd_test aihub_noisy_test' bash local/watch_bench.sh $EXP tiny2 12-15 >> exp-watch-pt5k.log 2>&1"
  fi
  sleep 60
done
echo "$(date +%m-%d_%H:%M) epoch $TOTAL done"
