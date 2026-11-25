#!/bin/bash
# One-shot VM bootstrap (run detached). Reuses the prebuilt gsplat CUDA library uploaded to /content/gsplat_cuda.so.
set -x
mkdir -p /content/hips /content/logs /content/out /content/data && cd /content/hips && tar xzf /content/hips_code.tgz
pip install -q gsplat plyfile 2>&1 | tail -1
# load the prebuilt CUDA library through a shim (skips the ~20 min JIT build on a 2-CPU VM)
G=$(python -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('gsplat').origin))")
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
if [ "$SM" = "75" ] && [ -f /content/gsplat_cuda.so ]; then cp /content/hips/remote/csrc_shim.py $G/csrc.py; fi
python -c "from gsplat.cuda._backend import _C; print('GSPLAT_READY', _C is not None)"
echo GSPLAT_DONE
# dataset (garden only) in the background of this script
( cd /content/data && wget -q http://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O 360_v2.zip && unzip -q 360_v2.zip 'garden/*' -d 360_v2 && echo DATA_DONE > data.done ) &
# trainer deps
cd /content && (test -d gsplat || git clone -q --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git)
touch /content/gsplat/examples/datasets/__init__.py
pip install -q viser "git+https://github.com/nerfstudio-project/nerfview" "imageio[ffmpeg]" scipy scikit-learn tqdm torchmetrics opencv-python-headless tyro piexif tensorboard tensorly pyyaml matplotlib splines 2>&1 | tail -1
pip uninstall -y -q pycolmap; pip install -q git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e 2>&1 | tail -1
sed -i 's/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/g' /usr/local/lib/python3.13/dist-packages/pycolmap/*.py
pip install -q --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim" 2>&1 | tail -1
python -c "import fused_ssim, pycolmap; print('DEPS_OK')"
wait
ls /content/data/360_v2/garden | head
echo ALL_DONE
