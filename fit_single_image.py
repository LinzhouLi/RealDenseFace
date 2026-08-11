"""Fit a FLAME mesh to a single image with RealDenseFace."""
from __future__ import annotations
import common.fix_chumpy  # noqa: F401  (must import before FLAME model load)

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import PerspectiveCamera
from tracker import (
    FaceBoxConfig,
    FLAMEConfig,
    GNFlameOptimizer,
    GNFlameOptimizerConfig,
    RealDenseFaceInferenceConfig,
    RealDenseFaceInferencer,
    load_fitting_config,
)
from visualization import Visualizer, render_reconstruction_panels, save_reconstruction_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image FLAME reconstruction with RealDenseFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a single image (jpg/png/...).")
    parser.add_argument("--output_npz", type=str, default="",
                        help="Output npz path for FLAME parameters. Empty means skip.")
    parser.add_argument("--output_image", type=str, default="",
                        help="Output 4-panel viz path (jpg/png). Empty means skip.")
    parser.add_argument("--fov_y_deg", type=float, default=None,
                        help="Override camera fov_y (deg). If unset, runs fov search inside fit_frame.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Mesh-on-image overlay blend weight.")

    parser.add_argument("--model_config", type=str, default="configs/model/vitb.yaml")
    parser.add_argument("--model_weights", type=str, default="weights/realdenseface/vitb.pth")
    parser.add_argument("--fitting_config", type=str, default="configs/fitting/single_image.yaml")
    parser.add_argument("--flame_model", type=str, default="weights/flame/flame2023.pkl")
    parser.add_argument("--flame_assets", type=str, default="weights/flame/flame_assets.npz")
    parser.add_argument("--facebox_weights", type=str, default="weights/facebox/face_box.pth")
    parser.add_argument("--num_expressions", type=int, choices=(50, 100), default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_compile", action="store_true")
    return parser.parse_args()


def ensure_parent_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_flame_npz(
    output_path: Path,
    fit_result,
    image_width: int,
    image_height: int,
    optimizer_config: GNFlameOptimizerConfig,
) -> None:
    camera_fov_y = np.float32(np.nan if fit_result.camera_fov_y is None else fit_result.camera_fov_y)
    np.savez(
        output_path,
        identity=fit_result.identity.detach().cpu().numpy().astype(np.float32),
        x=fit_result.x.detach().cpu().numpy().astype(np.float32),
        camera_fov_y=camera_fov_y,
        camera_rot=np.asarray(optimizer_config.camera_rotation, dtype=np.float32),
        camera_pos=np.asarray(optimizer_config.camera_position, dtype=np.float32),
        image_width=np.asarray(image_width, dtype=np.int32),
        image_height=np.asarray(image_height, dtype=np.int32),
        num_expressions=np.asarray(int(fit_result.x.shape[-1]) - 18, dtype=np.int32),
    )


def build_visualization_camera(
    fit_result,
    image_width: int,
    image_height: int,
    optimizer_config: GNFlameOptimizerConfig,
    fov_y_deg_override: float | None,
) -> PerspectiveCamera:
    if fov_y_deg_override is not None:
        fov_y_deg = float(fov_y_deg_override)
    elif fit_result.camera_fov_y is not None:
        fov_y_deg = float(fit_result.camera_fov_y)
    else:
        raise ValueError("Visualization needs a fov_y; pass --fov_y_deg or let fit_frame search it.")
    return PerspectiveCamera(
        fov_y=np.radians(fov_y_deg),
        rot=optimizer_config.camera_rotation,
        pos=optimizer_config.camera_position,
        width=image_width,
        height=image_height,
    )


def main() -> None:
    args = parse_args()

    output_npz_path = ensure_parent_dir(args.output_npz) if args.output_npz else None
    output_image_path = ensure_parent_dir(args.output_image) if args.output_image else None

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise ValueError(f"Cannot decode image: {input_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_height, image_width = image_rgb.shape[:2]
    print(f"[recon] input={input_path}  size=({image_width},{image_height})")

    flame_config = FLAMEConfig(
        flame_model_path=args.flame_model,
        flame_assets_path=args.flame_assets,
        num_expressions=int(args.num_expressions),
    )
    inferencer = RealDenseFaceInferencer(
        RealDenseFaceInferenceConfig(
            model_config_path=args.model_config,
            model_weights_path=args.model_weights,
            flame=flame_config,
            device=args.device,
            compile_model=not args.no_compile,
        ),
        FaceBoxConfig(model_weights_path=args.facebox_weights, device=args.device),
    )
    optimizer_config = load_fitting_config(
        args.fitting_config,
        GNFlameOptimizerConfig,
        flame=flame_config,
        device=args.device,
    )
    optimizer = GNFlameOptimizer(optimizer_config)

    start_t = time.perf_counter()

    inference = inferencer.infer_image(image_rgb, face_bbox=None)

    if args.fov_y_deg is not None:
        camera = PerspectiveCamera(
            fov_y=np.radians(float(args.fov_y_deg)),
            rot=optimizer_config.camera_rotation,
            pos=optimizer_config.camera_position,
            width=image_width,
            height=image_height,
        )
        fit_result = optimizer.fit_frame(inference, camera)
    else:
        fit_result = optimizer.fit_frame(inference, camera=None)

    elapsed_fit = time.perf_counter() - start_t
    print(f"[recon] fit_frame done in {elapsed_fit:.2f}s "
          f"(camera_fov_y={fit_result.camera_fov_y!r})")

    # FLAME params
    if output_npz_path is not None:
        save_flame_npz(output_npz_path, fit_result, image_width, image_height, optimizer_config)
        print(f"[recon] wrote npz: {output_npz_path}")

    vertices: torch.Tensor | None = None
    if output_image_path is not None:
        vertices, _ = optimizer.decode(fit_result)  # (5023, 3)

    # 4-panel viz
    if output_image_path is not None:
        visualizer = Visualizer(flame_assets_path=flame_config.flame_assets_path, device=args.device)
        camera = build_visualization_camera(
            fit_result, image_width, image_height, optimizer_config, args.fov_y_deg,
        )
        composite_rgb = render_reconstruction_panels(
            visualizer=visualizer,
            camera=camera,
            image_rgb=image_rgb,
            inference=inference,
            vertices=vertices,
            alpha=float(args.alpha),
        )
        save_reconstruction_image(output_image_path, composite_rgb)
        print(f"[recon] wrote viz: {output_image_path}")

    print(f"[recon] total runtime: {time.perf_counter() - start_t:.2f}s")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

