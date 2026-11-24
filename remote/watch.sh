#!/bin/bash
# Poll a VM log every 30 s; print new interesting lines; exit on ALL_DONE or a crash.
# usage: remote/watch.sh <logname> [regex]
cd "$(dirname "$0")/.."
RE="${2:-it +[0-9]+ loss|shift_ssim|Error|Traceback|ALL_DONE|Killed|out of memory|DONE}"
seen=0
while true; do
  out=$(bash remote/tail.sh "$1" 200000 2>/dev/null)
  n=$(printf '%s\n' "$out" | wc -l)
  if [ "$n" -gt "$seen" ]; then
    printf '%s\n' "$out" | tail -n +$((seen + 1)) | grep -E --line-buffered "$RE"
    seen=$n
  fi
  printf '%s\n' "$out" | grep -qE "ALL_DONE|Traceback" && exit 0
  sleep 30
done
