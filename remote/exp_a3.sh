#!/bin/bash
# Single-view (option A) with paintings: reveal-zone study (sharp vs. jitter-widened), plus evaluation videos.
cd /content/hips
while pgrep -f "simple_trainer|exp_tv3" > /dev/null; do sleep 20; done
for t in pearl_earring great_wave; do
  python -m hips.option_a --target targets/art_$t.png --out /content/out/a3_$t --n 400000 --steps 1200
  python -m hips.option_a --target targets/art_$t.png --out /content/out/a3_${t}_jit --n 400000 --steps 1500 --jitter 0.3
done
for r in a3_pearl_earring a3_pearl_earring_jit a3_great_wave a3_great_wave_jit; do
  t=${r#a3_}; t=${t%_jit}
  python -m hips.eval_reveal --scene /content/out/$r --target targets/art_$t.png --out /content/out/eval_$r --video
done
python -m hips.eval_reveal --scene /content/out/tv3_wave_starry --target targets/art_great_wave.png --out /content/out/eval_tv3_wave_starry --video
python -m hips.eval_reveal --scene /content/out/tv3_pearl_mona --target targets/art_pearl_earring.png --out /content/out/eval_tv3_pearl_mona --video
echo ALL_DONE
