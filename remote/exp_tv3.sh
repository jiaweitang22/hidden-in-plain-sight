#!/bin/bash
# Two-view paintings v2: size floor (no sub-pixel specks) + more specks + longer fit.
cd /content/hips
while pgrep -f simple_trainer > /dev/null; do sleep 20; done   # don't slow down the garden training
python -m hips.two_view --a targets/art_great_wave.png --b targets/art_starry_night.png --out /content/out/tv3_wave_starry --n 500000 --steps 1500 --title "The Wave and the Night" --label_a "The Great Wave" --label_b "The Starry Night"
python -m hips.two_view --a targets/art_pearl_earring.png --b targets/art_mona_lisa.png --out /content/out/tv3_pearl_mona --n 500000 --steps 1500 --title "Two Portraits" --label_a "Girl with a Pearl Earring" --label_b "Mona Lisa"
echo ALL_DONE
