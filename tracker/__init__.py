from __future__ import annotations

from typing import TYPE_CHECKING

from common.config import (
    FaceBoxConfig,
    FLAMEConfig,
    GNFlameOptimizerConfig,
    OnlineGNOptimizerConfig,
    RealDenseFaceInferenceConfig,
    load_fitting_config,
    load_model_config,
)
from common.types import FittingOutput, InferenceOutput

if TYPE_CHECKING:
    from .fitting.gn_flame_optimizer import GNFlameOptimizer
    from .fitting.online_gn_optimizer import OnlineGNOptimizer
    from .inference.realdenseface_inferencer import RealDenseFaceInferencer
    from .inference.video_input import VideoInput


_LAZY_IMPORTS = {
    "GNFlameOptimizer": (".fitting.gn_flame_optimizer", "GNFlameOptimizer"),
    "OnlineGNOptimizer": (".fitting.online_gn_optimizer", "OnlineGNOptimizer"),
    "RealDenseFaceInferencer": (
        ".inference.realdenseface_inferencer",
        "RealDenseFaceInferencer",
    ),
    "VideoInput": (".inference.video_input", "VideoInput"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "FaceBoxConfig",
    "FLAMEConfig",
    "GNFlameOptimizerConfig",
    "OnlineGNOptimizerConfig",
    "RealDenseFaceInferenceConfig",
    "load_fitting_config",
    "load_model_config",
    "FittingOutput",
    "InferenceOutput",
    "GNFlameOptimizer",
    "OnlineGNOptimizer",
    "RealDenseFaceInferencer",
    "VideoInput",
]
