#!/bin/bash
# Pin gsplat examples to the installed gsplat version and install the trainer's deps (without touching torch).
set -x
cd /content/gsplat && git fetch -q --depth 1 origin tag v1.5.3 && git checkout -q v1.5.3 && git log --oneline -1
cat examples/requirements.txt
pip install -q pycolmap viser "git+https://github.com/nerfstudio-project/nerfview" "imageio[ffmpeg]" scipy scikit-learn tqdm torchmetrics opencv-python-headless tyro piexif tensorboard tensorly pyyaml matplotlib splines 2>&1 | tail -3
pip install -q --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim" 2>&1 | tail -3
python -c "import torch, pycolmap, nerfview, fused_ssim; print('deps ok', torch.__version__)"
cd /content/gsplat/examples && python -c "import simple_trainer" && echo TRAINER_IMPORT_OK
echo ALL_DONE
