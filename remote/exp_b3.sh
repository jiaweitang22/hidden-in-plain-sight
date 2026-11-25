#!/bin/bash
# Option B v3: decal mode with grazing-angle + wider depth-cliff filtering (no streaks at the table rim).
cd /content/hips
while pgrep -f exp_b2.sh > /dev/null; do sleep 15; done
CK=$(ls /content/out/garden/ckpts/ckpt_*_rank0.pt | sort -V | tail -1)
D=/content/data/360_v2/garden; T=targets/art_pearl_earring_cut.png
python -m hips.option_b_decal --data $D --ckpt $CK --target $T --out /content/out/b3_el70 --elev 70 --azim 0 --shift 0 0.25
python -m hips.export_b --run /content/out/b3_el70 --out /content/web/b3_el70 --title "The Garden" --sh 1
python -m hips.eval_reveal --scene /content/web/b3_el70 --orig /content/web/b1_orig --target $T --out /content/out/eval_b3_el70 --frac 0.6 --video
echo ALL_DONE
