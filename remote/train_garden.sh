#!/bin/bash
# Train a compact garden splat (MCMC strategy caps the Gaussian count so it fits in a browser).
cd /content/gsplat/examples
python simple_trainer.py mcmc --data_dir /content/data/360_v2/garden --data_factor 4 \
  --result_dir /content/out/garden --disable_viewer --max_steps 20000 \
  --strategy.cap-max 600000 --save_steps 7000 20000 --eval_steps 20000
echo ALL_DONE
