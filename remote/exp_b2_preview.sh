#!/bin/bash
cd /content/hips
CK=$(ls /content/out/garden/ckpts/ckpt_*_rank0.pt | sort -V | tail -1)
python -m hips.option_b_decal --data /content/data/360_v2/garden --ckpt $CK --target targets/art_pearl_earring_cut.png --out /content/out/b2_preview --preview
echo ALL_DONE
