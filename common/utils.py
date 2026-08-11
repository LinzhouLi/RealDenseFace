from __future__ import annotations

import numpy as np

from common.config import FLAMEConfig


def load_mesh_from_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()
                faces.append([int(part.split("/")[0]) - 1 for part in parts[1:]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_expression_blendshapes(
    flame_pkl_data: dict,
    config: FLAMEConfig,
) -> np.ndarray:
    shapedirs = np.asarray(flame_pkl_data["shapedirs"])
    expr_dirs = shapedirs[..., 300:].astype(np.float32, copy=False)
    num_expressions = config.resolved_num_expressions()
    if expr_dirs.shape[-1] < num_expressions:
        raise ValueError(
            f"Requested {num_expressions} FLAME expression dims, "
            f"but model only has {expr_dirs.shape[-1]}."
        )
    return np.ascontiguousarray(expr_dirs[..., :num_expressions], dtype=np.float32)
