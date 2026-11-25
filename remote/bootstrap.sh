#!/bin/bash
# One-shot Colab VM setup (run detached after uploading /content/hips_code.tgz and, optionally,
# a prebuilt /content/gsplat_cuda.so for T4 / sm_75 -- see README "Running on a Colab GPU").
set -x
mkdir -p /content/hips /content/logs /content/out && cd /content/hips && tar xzf /content/hips_code.tgz
pip install -q gsplat plyfile "imageio[ffmpeg]" 2>&1 | tail -1
# Load a prebuilt CUDA library through a shim; skips gsplat's ~20 min JIT build on a 2-CPU VM.
G=$(python -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('gsplat').origin))")
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
if [ "$SM" = "75" ] && [ -f /content/gsplat_cuda.so ]; then cp /content/hips/remote/csrc_shim.py $G/csrc.py; fi
python -c "from gsplat.cuda._backend import _C; print('GSPLAT_READY', _C is not None)"
echo ALL_DONE
