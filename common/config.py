from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml


INVISIBLE_LOG_VAR: float = 30.0


@dataclass(slots=True)
class FLAMEConfig:
    flame_model_path: str = "weights/flame/flame2023.pkl"
    flame_assets_path: str = "weights/flame/flame_assets.npz"
    num_expressions: int = 100

    def resolved_num_expressions(self) -> int:
        num_expressions = int(self.num_expressions)
        if num_expressions not in (50, 100):
            raise ValueError(
                f"FLAMEConfig.num_expressions only supports 50 or 100, got {num_expressions}."
            )
        return num_expressions


@dataclass(slots=True)
class RealDenseFaceInferenceConfig:
    model_config_path: str = "configs/model/vitb.yaml"
    model_weights_path: str = "weights/realdenseface/vitb.pth"
    flame: FLAMEConfig = field(default_factory=FLAMEConfig)
    device: str = "cuda"
    enlarge_bbox_ratio: float = 1.3
    compile_model: bool = True
    output_type: str = "numpy"
    filter_out_of_bounds_vertices: bool = False
    filter_occluded_vertices: bool = False
    filter_invisible_log_var: float = INVISIBLE_LOG_VAR
    filter_visibility_depth_eps: float = 0.1


@dataclass(slots=True)
class FaceBoxConfig:
    model_weights_path: str = "weights/facebox/face_box.pth"
    device: str = "cuda"
    score_threshold: float = 0.6
    input_width: int = 512
    input_height: int = 512
    half: bool = True


@dataclass(slots=True)
class GNFlameOptimizerConfig:
    flame: FLAMEConfig = field(default_factory=FLAMEConfig)
    device: str = "cuda"
    use_uncertainty_weights: bool = True
    correspondence_weight: float = 1.0
    relative_depth_weight: float = 2.0
    expression_regularization: float = 1e-2
    pose_regularization: float = 1e-2
    identity_regularization: float = 1e-2
    damping: float = 1e-3
    pose_damping: float = 0.1
    registration_iterations: int = 20
    pose_registration_iterations: int = 10
    tracking_iterations: int = 20
    root_pose_tracking_iterations: int = 3
    tracking_early_stop_ratio: float = 2e-4
    identity_refinement_rounds: int = 2
    num_keyframes: int = 32
    energy_chunk_size: int = 256
    camera_rotation: np.ndarray = field(
        default_factory=lambda: np.diag(np.array([1.0, -1.0, -1.0], dtype=np.float32))
    )
    camera_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    fov_y_min: float = 20.0
    fov_y_max: float = 60.0
    fov_search_iterations: int = 5
    fov_search_epsilon: float = 0.1
    fov_registration_iterations: int = 15
    boost_eye_region: bool = True
    eye_region_boost_factor: float = 10.0


@dataclass(slots=True)
class OnlineGNOptimizerConfig:
    flame: FLAMEConfig = field(default_factory=FLAMEConfig)
    device: str = "cuda"
    correspondence_weight: float = 1.0
    relative_depth_weight: float = 0.0
    expression_regularization: float = 1e-2
    pose_regularization: float = 1e-2
    identity_regularization: float = 1e-2
    damping: float = 1e-3
    pose_damping: float = 0.1
    registration_iterations: int = 10
    pose_registration_iterations: int = 10
    tracking_iterations: int = 10
    tracking_early_stop_ratio: float = 1e-4
    keyframe_interval: int = 10
    max_keyframes: int = 12
    identity_refinement_steps: int = 4
    keyframe_head_threshold: float = 0.288
    fov_y_min: float = 25.0
    fov_y_max: float = 60.0
    fov_search_iterations: int = 5
    fov_search_epsilon: float = 0.1
    fov_registration_iterations: int = 10
    boost_eye_region: bool = True
    eye_region_boost_factor: float = 10.0


ConfigT = TypeVar("ConfigT", GNFlameOptimizerConfig, OnlineGNOptimizerConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}.")
    return data


def load_model_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    required = {"target_size", "dino_encoder", "scale_coord", "scale_depth"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Model config {path} is missing required fields: {missing}")
    return config


def load_fitting_config(
    path: str | Path,
    config_type: type[ConfigT],
    *,
    flame: FLAMEConfig,
    device: str,
) -> ConfigT:
    values = load_yaml(path)
    allowed = {item.name for item in fields(config_type)} - {"flame", "device"}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported fields in fitting config {path}: {unknown}")

    for name in ("camera_rotation", "camera_position"):
        if name in values:
            values[name] = np.asarray(values[name], dtype=np.float32)

    return config_type(flame=flame, device=device, **values)
