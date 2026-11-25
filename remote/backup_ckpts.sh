#!/bin/bash
# Copy new garden checkpoints from the VM to ./cache/garden_ckpts as they appear (run in background).
cd "$(dirname "$0")/.." && mkdir -p cache/garden_ckpts
while true; do
  for f in $(timeout 100 colab ls -s hips /content/out/garden/ckpts 2>/dev/null | grep -o "ckpt_[0-9]*_rank0.pt"); do
    [ -f cache/garden_ckpts/$f ] || { timeout 900 colab download -s hips /content/out/garden/ckpts/$f cache/garden_ckpts/$f >/dev/null 2>&1 && echo "backed up $f"; }
  done
  ls cache/garden_ckpts | grep -q ckpt_19999 && exit 0
  sleep 120
done
