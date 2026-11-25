#!/bin/bash
# Three paintings in one cloud: front, side (90 deg), and from above (80 deg up).
cd /content/hips
while pgrep -f "simple_trainer|exp_tv3" > /dev/null; do sleep 20; done
python -m hips.multi_view --targets targets/art_great_wave.png targets/art_starry_night.png targets/art_pearl_earring.png \
  --views 0,0 90,0 0,-80 --labels "The Great Wave" "The Starry Night" "Girl with a Pearl Earring" \
  --out /content/out/mv3_art --title "Three Paintings" --n 600000
echo ALL_DONE
