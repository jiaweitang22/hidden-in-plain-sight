#!/bin/bash
# Studies on multi-painting clouds (metrics + contact sheets only; no PLY).
cd /content/hips
while pgrep -f "exp_mv1|exp_a3" > /dev/null; do sleep 20; done
W=targets/art_great_wave.png; S=targets/art_starry_night.png; P=targets/art_pearl_earring.png; M=targets/art_mona_lisa.png
C="--n 500000 --steps 1000 --no_ply"
# 1) separation: how close can two secret spots be?
for ang in 90 60 45 30 15; do
  python -m hips.multi_view --targets $W $S --views 0,0 $ang,0 --out /content/out/study/sep_$ang $C
done
# 2) colour model: same colour from every angle (SH0) vs. angle-dependent colour (SH1, SH2)
for sh in 0 2; do
  python -m hips.multi_view --targets $W $S --views 0,0 90,0 --out /content/out/study/sh_$sh $C --sh $sh
done
# 3) capacity: 1, 2, 3, 4 paintings in one cloud (same speck budget)
python -m hips.multi_view --targets $W --views 0,0 --out /content/out/study/cap_1 $C
python -m hips.multi_view --targets $W $S $P --views 0,0 90,0 0,-80 --out /content/out/study/cap_3 $C
python -m hips.multi_view --targets $W $S $M $P --views 0,0 90,0 180,0 0,-80 --out /content/out/study/cap_4 $C
python -m hips.multi_view --targets $W $S $M $P --views 0,0 90,0 180,0 0,-80 --out /content/out/study/cap_4_sh2 $C --sh 2
# 4) speck budget
for n in 100000 250000 1000000; do
  python -m hips.multi_view --targets $W $S --views 0,0 90,0 --out /content/out/study/n_$n --n $n --steps 1000 --no_ply
done
echo ALL_DONE
