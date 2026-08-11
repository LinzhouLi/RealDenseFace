from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class InferenceOutput:
    vertex_coord: np.ndarray | torch.Tensor
    vertex_coord_log_var: np.ndarray | torch.Tensor
    vertex_depth: np.ndarray | torch.Tensor
    vertex_depth_log_var: np.ndarray | torch.Tensor
    face_bbox: np.ndarray
    image_width: int
    image_height: int
    frame_valid: np.ndarray | torch.Tensor | None = None


@dataclass(slots=True)
class OnlineInferenceOutput:
    vertex_coord: torch.Tensor
    vertex_coord_log_var: torch.Tensor
    vertex_depth: torch.Tensor
    vertex_depth_log_var: torch.Tensor
    face_bbox: np.ndarray
    image_width: int
    image_height: int
    frame_rgba: torch.Tensor


@dataclass(slots=True)
class FittingOutput:
    identity: torch.Tensor
    x: torch.Tensor
    pose_x: torch.Tensor | None = None
    camera_fov_y: float | None = None
    per_frame_views: list[list[int]] | None = None
    frame_reliable: np.ndarray | torch.Tensor | None = None
    register_keyframes: list | None = None
