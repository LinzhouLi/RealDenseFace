"""Inference adapters shared by the Gradio app and local verification."""
import gc
from pathlib import Path
import time

import common.fix_chumpy  # noqa: F401
import numpy as np
import torch

from camera import PerspectiveCamera
from common.config import (FLAMEConfig, FaceBoxConfig, RealDenseFaceInferenceConfig,
                           GNFlameOptimizerConfig, load_fitting_config)
from tracker import RealDenseFaceInferencer, GNFlameOptimizer
from visualization import Visualizer


class ReconstructionEngine:
    def __init__(self, root: Path, model_weights: str, facebox_weights: str, flame_assets: str, flame_model: str):
        self.root = root
        self.flame = FLAMEConfig(flame_model_path=flame_model, flame_assets_path=flame_assets)
        self.inferencer = RealDenseFaceInferencer(
            RealDenseFaceInferenceConfig(
                model_config_path=str(root / "configs/model/vitb.yaml"),
                model_weights_path=model_weights, flame=self.flame,
                compile_model=False, defer_native_context=True,
            ), FaceBoxConfig(model_weights_path=facebox_weights),
        )
        self.fitting_config = load_fitting_config(root / "configs/fitting/single_image.yaml", GNFlameOptimizerConfig,
                                                 flame=self.flame, device="cuda")

    @torch.inference_mode()
    def reconstruct(self, image: np.ndarray):
        started = time.perf_counter()
        # Preserve the official pipeline's bbox refinement and fallback: some
        # heavily occluded teaser faces are missed by the separate detector.
        inference = self.inferencer.infer_image(image, face_bbox=None)
        optimizer = None
        visualizer = None
        try:
            optimizer = GNFlameOptimizer(self.fitting_config)
            fit = optimizer.fit_frame(inference, camera=None)
            vertices, _ = optimizer.decode(fit)
            if not torch.isfinite(vertices).all().item():
                raise RuntimeError("Fitting produced non-finite vertices")
            camera = PerspectiveCamera(
                fov_y=np.radians(float(fit.camera_fov_y)),
                rot=self.fitting_config.camera_rotation, pos=self.fitting_config.camera_position,
                width=image.shape[1], height=image.shape[0],
            )
            visualizer = Visualizer(flame_assets_path=self.flame.flame_assets_path)
            dense = visualizer.vis_align(image, inference.vertex_coord)
            dense = visualizer.vis_contour(dense, inference.vertex_coord)
            overlay, mesh = visualizer.vis_flame_shading_overlay(camera, image, vertices, return_shading=True)
            torch.cuda.synchronize()
            return dense, overlay, mesh, {
                "seconds": round(time.perf_counter() - started, 3),
                "gpu": torch.cuda.get_device_name(0),
                "fov_y": float(fit.camera_fov_y),
                "vertices": int(vertices.shape[0]),
            }
        finally:
            torch.cuda.synchronize()
            if optimizer is not None:
                ops = optimizer.solver_ops.extension
                ops.destroy_cublas_handle(optimizer.solver_context.cublas_handle)
                ops.destroy_cusolver_handle(optimizer.solver_context.cusolver_handle)
            del optimizer, visualizer
            gc.collect()
