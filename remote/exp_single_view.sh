#!/bin/bash
# Single-view clouds (option A) of two paintings: reveal-zone study (sharp vs. jitter-widened) + fly-in videos.
cd /content/hips
for t in pearl_earring great_wave; do
  python -m hips.option_a --target targets/art_$t.png --out /content/out/a3_$t --n 400000 --steps 1200
  python -m hips.option_a --target targets/art_$t.png --out /content/out/a3_${t}_jit --n 400000 --steps 1500 --jitter 0.3
done
for r in a3_pearl_earring a3_pearl_earring_jit a3_great_wave a3_great_wave_jit; do
  t=${r#a3_}; t=${t%_jit}
  python -m hips.eval_reveal --scene /content/out/$r --target targets/art_$t.png --out /content/out/eval_$r --video
done
for r in mv3_art tv_wave_starry; do  # fly-in videos of gallery scenes (spot 1); run exp_gallery.sh first
  [ -d /content/out/$r ] && python -m hips.eval_reveal --scene /content/out/$r --target targets/art_great_wave.png --out /content/out/eval_$r --video
done
echo ALL_DONE
