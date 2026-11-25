# Installed as gsplat/csrc.py: load the prebuilt gsplat CUDA extension without the JIT build path.
import importlib.util
import torch  # noqa: F401  (registers torch symbols the extension needs)
_spec = importlib.util.spec_from_file_location("gsplat_cuda", "/content/gsplat_cuda.so")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
globals().update({k: v for k, v in _m.__dict__.items() if not k.startswith("__")})
