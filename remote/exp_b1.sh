#!/bin/bash
# Experiment B1: hide the eye in the garden. Compare projection painting alone vs. painting + optimization
# vs. optimization from scratch. Then export + evaluate each like the viewer sees it.
cd /content/hips
CK=$(ls /content/out/garden/ckpts/ckpt_*_rank0.pt | sort -V | tail -1)
echo "checkpoint $CK"
D=/content/data/360_v2/garden
python -m hips.option_b --data $D --ckpt $CK --target targets/eye.png --out /content/out/b1_project --steps 0
python -m hips.option_b --data $D --ckpt $CK --target targets/eye.png --out /content/out/b1_proj_opt --steps 2000
python -m hips.option_b --data $D --ckpt $CK --target targets/eye.png --out /content/out/b1_opt_only --steps 2000 --init none
for r in b1_project b1_proj_opt b1_opt_only; do
  python -m hips.export_b --run /content/out/$r --out /content/web/$r --title "Garden: $r" --sh 1
  python -m hips.eval_reveal --scene /content/web/$r --target targets/eye.png --out /content/out/eval_$r --frac 0.6 --video
done
echo ALL_DONE
