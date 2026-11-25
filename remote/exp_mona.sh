#!/bin/bash
# Single-painting gallery scene (the intro): Mona Lisa.
cd /content/hips
python -m hips.multi_view --targets targets/art_mona_lisa.png --views 0,0 --n 400000 --steps 1500 \
  --out /content/out/one_mona --title "Mona Lisa" --labels "Mona Lisa"
echo ALL_DONE
