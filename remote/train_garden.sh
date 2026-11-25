#!/bin/bash
# Train a compact garden splat (MCMC caps the Gaussian count so it can load in a browser).
cd /content/gsplat/examples
python -u simple_trainer.py mcmc --data_dir /content/data/360_v2/garden --data_factor 4 \
  --result_dir /content/out/garden --disable_viewer --max_steps 20000 \
  --strategy.cap-max 600000 --save_steps 5000 10000 15000 20000 --eval_steps 20000 2>&1 \
  | tr '\r' '\n' | grep --line-buffered -E "Step|PSNR|Error|error|Traceback|Saving|psnr" | grep --line-buffered -vE "it/s\]$" | tail -n 200 > /content/logs/garden_filtered.log
echo ALL_DONE
