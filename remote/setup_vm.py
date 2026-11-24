# Run on the Colab VM: install deps, unpack code, start dataset download + gsplat clone in background.
import subprocess
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print(r.stdout[-2500:], r.stderr[-1500:])
sh("mkdir -p /content/hips && cd /content/hips && tar xzf /content/hips_code.tgz && ls")
sh("pip install -q gsplat plyfile 2>&1 | tail -2; pip show gsplat | head -2")
sh("mkdir -p /content/data && cd /content/data && nohup sh -c 'wget -q http://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O 360_v2.zip && unzip -q 360_v2.zip -d 360_v2 && echo DONE > download.done' > download.log 2>&1 &")
sh("cd /content && (test -d gsplat || git clone -q --depth 1 https://github.com/nerfstudio-project/gsplat.git) && ls gsplat/examples | head -20")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python -c 'import torch;print(torch.__version__)'")
