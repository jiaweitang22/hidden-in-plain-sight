#!/bin/bash
# Poll a VM log every 30 s; print new interesting lines; exit on ALL_DONE or a crash in NEW output.
# usage: remote/watch.sh <logname> [regex] [--from-start]
cd "$(dirname "$0")/.."
RE="${2:-it +[0-9]+ loss|shift_ssim|RESULT|Error|Traceback|ALL_DONE|Killed|out of memory|DONE}"
if [ "$3" = "--from-start" ]; then seen=0; else seen=$(bash remote/tail.sh "$1" 400000 2>/dev/null | wc -l); fi
while true; do
  out=$(bash remote/tail.sh "$1" 400000 2>/dev/null)
  n=$(printf '%s\n' "$out" | wc -l)
  if [ "$n" -gt "$seen" ]; then
    new=$(printf '%s\n' "$out" | tail -n +$((seen + 1)))
    printf '%s\n' "$new" | grep -E --line-buffered "$RE"
    seen=$n
    printf '%s\n' "$new" | grep -qE "ALL_DONE|Traceback" && exit 0
  fi
  sleep 30
done
