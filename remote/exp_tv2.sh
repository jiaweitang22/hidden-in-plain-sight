#!/bin/bash
# Two-view anamorphosis with public-domain paintings. Waits for exp_tv1 to finish.
cd /content/hips
while pgrep -f exp_tv1.sh > /dev/null; do sleep 10; done
python -m hips.two_view --a targets/art_great_wave.png --b targets/art_starry_night.png --out /content/out/tv_wave_starry --n 320000 --title "The Wave and the Night" --label_a "The Great Wave" --label_b "The Starry Night"
python -m hips.two_view --a targets/art_pearl_earring.png --b targets/art_mona_lisa.png --out /content/out/tv_pearl_mona --n 320000 --title "Two Portraits" --label_a "Girl with a Pearl Earring" --label_b "Mona Lisa"
echo ALL_DONE
