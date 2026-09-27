#!/usr/bin/env bash
# Run local/kws_bench.sh (CER + KWS on the held-out sets) for every new epoch-N.pt, N >= 2, avg 1.
# Pinned to a few CPU cores so the training data loader keeps its CPUs.
#   local/watch_bench.sh <exp> <arch> [cores=12-15]      (SETS env as in kws_bench.sh)
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"; cd "$KWS_TRAIN"
exp=$1; arch=$2; cores=${3:-12-15}
while true; do
  systemctl --user is-active -q "${UNIT:-kws-pt5k}"; active=$?
  for ck in $(ls "$exp"/epoch-*.pt 2>/dev/null | sed 's/.*epoch-//; s/.pt//' | sort -n); do
    [ "$ck" -ge 2 ] || continue
    [ -f "$exp/bench-e$ck-a1${OUT_SUFFIX:-}/report.txt" ] && continue
    sleep 30   # let the checkpoint finish writing
    taskset -c "$cores" bash local/kws_bench.sh "$exp" "$arch" "$ck" 1 > "$exp/bench-e$ck.log" 2>&1
    echo "$(date +%m-%d_%H:%M) epoch $ck done"
  done
  # training unit finished and every epoch benched → stop
  [ $active -eq 0 ] || exit 0
  sleep 120
done
