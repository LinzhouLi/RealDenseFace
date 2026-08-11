import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_EXTENSIONS_DIR = os.path.dirname(ROOT_DIR)
SHARED_GLM_DIR = os.path.normpath(os.path.join(CUDA_EXTENSIONS_DIR, "third_party", "glm"))

if not os.path.isdir(SHARED_GLM_DIR):
    raise RuntimeError(f"Shared GLM dir not found: {SHARED_GLM_DIR}")

if sys.platform == "win32":
    cxx_flags = ["/O2"]
    libraries = ["cublas", "cusolver"]
else:
    cxx_flags = ["-O3"]
    libraries = ["cublas", "cusolver"]

setup(
    name="flame_solver",
    packages=["flame_solver"],
    package_dir={"": "."},
    ext_modules=[
        CUDAExtension(
            name="flame_solver.cuda_ext",
            sources=[
                "src/cholesky.cu",
                "src/assemble_jacobian.cu",
                "src/assemble_direct_jacobian.cu",
                "src/flame_expression_jacobian.cu",
                "src/flame_identity_jacobian.cu",
                "src/flame_calc_canonical.cu",
                "src/pcg.cu",
                "binding.cpp",
            ],
            include_dirs=[SHARED_GLM_DIR],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": ["-O3", "-I" + SHARED_GLM_DIR, "-allow-unsupported-compiler"],
            },
            libraries=libraries
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
