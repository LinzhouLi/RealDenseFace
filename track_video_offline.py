from __future__ import annotations
import common.fix_chumpy  # noqa: F401  (must import before FLAME model load)

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from camera import PerspectiveCamera
from tracker import (
    FLAMEConfig,
    FaceBoxConfig,
    GNFlameOptimizer,
    GNFlameOptimizerConfig,
    RealDenseFaceInferenceConfig,
    RealDenseFaceInferencer,
    VideoInput,
    load_fitting_config,
)
from visualization import Visualizer, save_tracking_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a monocular video or frame directory and optionally save results.")
    parser.add_argument("--input", type=str, required=True, help="Path to a video file or a frame directory.")
    parser.add_argument("--output_npz", type=str, default="", help="Output npz path for FLAME parameters. Empty means skip.")
    parser.add_argument("--output_mp4", type=str, default="", help="Output mp4 path for visualization. Empty means skip.")
    parser.add_argument("--fps", type=float, default=25.0, help="Fallback fps for visualization when the input has no fps.")
    parser.add_argument("--model_config", type=str, default="configs/model/vitb.yaml")
    parser.add_argument("--model_weights", type=str, default="weights/realdenseface/vitb.pth")
    parser.add_argument("--fitting_config", type=str, default="configs/fitting/video_offline.yaml")
    parser.add_argument("--flame_model", type=str, default="weights/flame/flame2023.pkl")
    parser.add_argument("--flame_assets", type=str, default="weights/flame/flame_assets.npz")
    parser.add_argument("--facebox_weights", type=str, default="weights/facebox/face_box.pth")
    parser.add_argument("--num_expressions", type=int, choices=(50, 100), default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--discontinuous", action="store_true")
    return parser.parse_args()


def ensure_parent_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_flame_npz(
    output_path: Path,
    fit_result,
    inference,
    optimizer_config: GNFlameOptimizerConfig,
) -> None:
    camera_fov_y = np.nan if fit_result.camera_fov_y is None else np.float32(fit_result.camera_fov_y)
    np.savez(
        output_path,
        identity=fit_result.identity.detach().cpu().numpy().astype(np.float32),
        x=fit_result.x.detach().cpu().numpy().astype(np.float32),
        camera_fov_y=np.asarray(camera_fov_y, dtype=np.float32),
        camera_rot=np.asarray(optimizer_config.camera_rotation, dtype=np.float32),
        camera_pos=np.asarray(optimizer_config.camera_position, dtype=np.float32),
        image_width=np.asarray(inference.image_width, dtype=np.int32),
        image_height=np.asarray(inference.image_height, dtype=np.int32),
        num_expressions=np.asarray(int(fit_result.x.shape[-1]) - 18, dtype=np.int32),
    )


def build_visualization_camera(
    fit_result,
    inference,
    optimizer_config: GNFlameOptimizerConfig,
) -> PerspectiveCamera:
    if fit_result.camera_fov_y is None:
        raise ValueError("Visualization requires camera_fov_y, but the fit result does not contain it.")
    return PerspectiveCamera(
        fov_y=np.radians(fit_result.camera_fov_y),
        rot=optimizer_config.camera_rotation,
        pos=optimizer_config.camera_position,
        width=inference.image_width,
        height=inference.image_height,
    )


def main() -> None:
    args = parse_args()
    if args.output_npz == "" and args.output_mp4 == "":
        raise ValueError("At least one of --output_npz or --output_mp4 must be non-empty.")

    output_npz_path = ensure_parent_dir(args.output_npz) if args.output_npz != "" else None
    output_mp4_path = ensure_parent_dir(args.output_mp4) if args.output_mp4 != "" else None

    video_input = VideoInput(args.input)
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

    start_time = time.perf_counter()
    inference = inferencer.infer_video(video_input, discontinuous=args.discontinuous)
    fit_result = optimizer.fit_sequence(inference, camera=None)
    elapsed = time.perf_counter() - start_time

    if output_npz_path is not None:
        save_flame_npz(output_npz_path, fit_result, inference, optimizer_config)
        print(f"Saved npz: {output_npz_path}")

    if output_mp4_path is not None:
        camera = build_visualization_camera(fit_result, inference, optimizer_config)
        visualizer = Visualizer(
            flame_assets_path=flame_config.flame_assets_path,
            device=args.device,
        )
        vertices, _ = optimizer.decode(fit_result)
        save_tracking_video(
            output_mp4_path,
            video_input,
            inference,
            camera,
            visualizer,
            vertices,
            args.fps,
        )
        print(f"Saved mp4: {output_mp4_path}")

    video_input.close()
    num_frames = int(inference.vertex_coord.shape[0])
    print(f"Total runtime: {elapsed:.3f}s, num_frames: {num_frames}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

