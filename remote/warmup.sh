#!/bin/bash
# Build gsplat's CUDA extension once, alone, with limited parallelism.
pkill -f hips.option_a; pkill -f ninja; pkill -f nvcc; pkill -f setup_train.sh; sleep 2
rm -rf ~/.cache/torch_extensions
export MAX_JOBS=3
python - <<'PY'
import time, torch
t = time.time()
from gsplat import rasterization
m = torch.zeros(10, 3, device="cuda"); m[:, 2] = 3
q = torch.zeros(10, 4, device="cuda"); q[:, 0] = 1
out, _, _ = rasterization(m, q, torch.full((10, 3), .1, device="cuda"), torch.ones(10, device="cuda"),
                          torch.rand(10, 3, device="cuda"), torch.eye(4, device="cuda")[None],
                          torch.tensor([[[100., 0, 32], [0, 100., 32], [0, 0, 1]]], device="cuda"), 64, 64)
print("GSPLAT_OK", out.shape, f"{time.time() - t:.0f}s", flush=True)
PY
echo ALL_DONE
