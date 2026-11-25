#!/bin/bash
cd /content/hips
for t in eye text smiley; do python -m hips.eval_reveal --scene /content/out/a2_$t --target targets/$t.png --out /content/out/eval_a2_$t --video; done
python -m hips.eval_reveal --scene /content/out/a2_eye_jit --target targets/eye.png --out /content/out/eval_a2_eye_jit --video
echo ALL_DONE
