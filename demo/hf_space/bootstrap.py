"""Compile the project's native extension before accepting GPU requests."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def prepare_solver():
    # chumpy's legacy setup imports pip, which is absent in pip's isolated
    # build environment. Install it on CPU with isolation disabled instead.
    if importlib.util.find_spec("chumpy") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "chumpy==0.70"],
            check=True, timeout=180,
        )
    if importlib.util.find_spec("flame_solver") is not None:
        return
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    print(f"[build] Python={sys.version} torch={torch.__version__} CUDA={torch.version.cuda} CUDA_HOME={CUDA_HOME} nvcc={shutil.which('nvcc')}", flush=True)
    if CUDA_HOME is None:
        raise RuntimeError("CUDA development toolkit unavailable: cannot build flame_solver in this runtime")
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "12.0+PTX"
    env["MAX_JOBS"] = "2"
    source = Path(__file__).resolve().parent / "cuda_extensions/flame_solver"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(source)],
        env=env, check=True, timeout=1800,
    )
