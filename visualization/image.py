from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from camera import Camera
from tracker.inference.preprocess import enlarge_bbox

from .visualizer import Visualizer


def render_reconstruction_panels(
    visualizer: Visualizer,
    camera: Camera,
    image_rgb: np.ndarray,
    inference,
    vertices: torch.Tensor,
    alpha: float = 0.5,
) -> np.ndarray:
    alignment = visualizer.vis_align(image_rgb, inference.vertex_coord)
    alignment = visualizer.vis_contour(alignment, inference.vertex_coord)
    bbox = np.round(enlarge_bbox(inference.face_bbox, 1.3)).astype(np.int32)
    cv2.rectangle(alignment, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), 2)
    overlay, shading = visualizer.vis_flame_shading_overlay(
        camera, image_rgb, vertices, alpha=alpha, return_shading=True
    )
    return np.ascontiguousarray(
        np.concatenate(
            [
                np.concatenate([image_rgb, alignment], axis=1),
                np.concatenate([overlay, shading], axis=1),
            ],
            axis=0,
        )
    )


def save_reconstruction_image(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write visualization image: {path}")

