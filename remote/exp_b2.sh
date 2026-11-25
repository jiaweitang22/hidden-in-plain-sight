#!/bin/bash
# Option B v2 (decal / Varini mode): Girl with a Pearl Earring painted into the garden, two candidate spots.
cd /content/hips
CK=$(ls /content/out/garden/ckpts/ckpt_*_rank0.pt | sort -V | tail -1)
D=/content/data/360_v2/garden; T=targets/art_pearl_earring_cut.png
python -m hips.option_b_decal --data $D --ckpt $CK --target $T --out /content/out/b2_el70 --elev 70 --azim 0 --shift 0 0.25
python -m hips.option_b_decal --data $D --ckpt $CK --target $T --out /content/out/b2_el50 --elev 50 --azim 0 --shift 0 0.25
while [ ! -f /content/web/b1_orig/scene.ply ]; do sleep 20; done
for r in b2_el70 b2_el50; do
  python -m hips.export_b --run /content/out/$r --out /content/web/$r --title "The Garden" --sh 1
  python -m hips.eval_reveal --scene /content/web/$r --orig /content/web/b1_orig --target $T --out /content/out/eval_$r --frac 0.6 --video
done
echo ALL_DONE
