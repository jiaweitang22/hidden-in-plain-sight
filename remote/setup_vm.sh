#!/bin/bash
# Colab VM setup. Run detached: nohup bash setup_vm.sh > setup.log 2>&1 &
set -x
mkdir -p /content/hips && cd /content/hips && tar xzf /content/hips_code.tgz
pip install -q gsplat plyfile 2>&1 | tail -2
python -c "import gsplat; print('gsplat', gsplat.__version__)"
cd /content && (test -d gsplat || git clone -q --depth 1 https://github.com/nerfstudio-project/gsplat.git)
echo SETUP_DONE
mkdir -p /content/data && cd /content/data
wget -q http://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O 360_v2.zip && unzip -q 360_v2.zip -d 360_v2 && rm 360_v2.zip && echo DATA_DONE
