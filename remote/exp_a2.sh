#!/bin/bash
# Experiment A2: "ball" placement (round dust cloud from other angles) for 3 targets + a jittered fit.
cd /content/hips
for t in eye text smiley; do
  python -m hips.option_a --target targets/$t.png --out /content/out/a2_$t --steps 800
done
python -m hips.option_a --target targets/eye.png --out /content/out/a2_eye_jit --steps 1200 --jitter 0.3
echo ALL_DONE
