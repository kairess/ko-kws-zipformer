#!/usr/bin/env bash
# Run a long job in its own systemd --user unit: survives Claude Code / tmux exits, and a memory
# cap (MemoryMax) makes the OOM killer take only this job instead of the whole session.
#   local/job.sh <name> <mem e.g. 24G> <command...>
# Status: systemctl --user status kws-<name> ; stop: systemctl --user stop kws-<name>
name=$1; mem=$2; shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
systemctl --user reset-failed "kws-$name" 2>/dev/null
exec systemd-run --user --unit="kws-$name" --collect -p MemoryMax="$mem" -p MemorySwapMax=2G \
  --working-directory="$PWD" -E PATH="$PATH" -E HOME="$HOME" \
  bash -lc "$*"
