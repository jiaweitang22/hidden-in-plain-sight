#!/bin/bash
# Experiment A1: off-view go/no-go across target types, plus jittered fit.
cd /content/hips
for t in smiley eye text; do
  python -m hips.option_a --target targets/$t.png --out /content/out/a1_$t --steps 1000
done
python -m hips.option_a --target targets/eye.png --out /content/out/a1_eye_jit --steps 1500 --jitter 0.3
echo ALL_DONE
