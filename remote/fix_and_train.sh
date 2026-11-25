#!/bin/bash
# Make gsplat v1.5.3 examples work on Colab (py3.13, numpy 2), then train a compact garden splat.
set -x
F=/usr/local/lib/python3.13/dist-packages/pycolmap
sed -i 's/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/g' $F/*.py
grep -rn "uint64(" $F | head -5
touch /content/gsplat/examples/datasets/__init__.py
cd /content/gsplat/examples
python simple_trainer.py mcmc --help 2>&1 | grep -iE "cap.max|data.dir|data.factor|save.steps|eval.steps|viewer|max.steps|ply" | head -20
python simple_trainer.py mcmc --data_dir /content/data/360_v2/garden --data_factor 4 \
  --result_dir /content/out/garden --disable_viewer --max_steps 20000 \
  --strategy.cap-max 600000 --save_steps 7000 20000 --eval_steps 20000 2>&1 | tr '\r' '\n' | grep -vE "^\s*$" | awk 'NR%50==1 || /Error|error|Traceback|PSNR|Step|ckpt|Saving/'
echo ALL_DONE
