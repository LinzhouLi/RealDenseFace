from __future__ import annotations

import common.fix_chumpy  # noqa: F401  (must import before FLAME model load)

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from camera import PerspectiveCamera
from tracker import (
    FLAMEConfig,
    FaceBoxConfig,
    OnlineGNOptimizer,
    OnlineGNOptimizerConfig,
    RealDenseFaceInferenceConfig,
    RealDenseFaceInferencer,
    VideoInput,
    load_fitting_config,
)
from tracker.inference.preprocess import enlarge_bbox
from visualization import Visualizer


def build_camera(
    image_width: int,
    image_height: int,
    fov_y_deg: float,
    camera_distance: float,
    znear: float,
    zfar: float,
) -> PerspectiveCamera:
    return PerspectiveCamera(
        fov_y=np.radians(float(fov_y_deg)),
        rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        pos=np.array([0.0, 0.0, float(camera_distance)], dtype=np.float32),
        width=int(image_width),
        height=int(image_height),
        znear=float(znear),
        zfar=float(zfar),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RealDenseFace online tracking on a monocular video."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output_mp4",
        type=Path,
        default=None,
        help="Output 2x2 tracking visualization. Empty means skip.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback output FPS when the input video has no FPS metadata.",
    )
    parser.add_argument("--model_config", type=str, default="configs/model/vits.yaml")
    parser.add_argument("--model_weights", type=str, default="weights/realdenseface/vits.pth")
    parser.add_argument("--fitting_config", type=str, default="configs/fitting/video_online.yaml")
    parser.add_argument("--flame_model", type=str, default="weights/flame/flame2023.pkl")
    parser.add_argument("--flame_assets", type=str, default="weights/flame/flame_assets.npz")
    parser.add_argument("--facebox_weights", type=str, default="weights/facebox/face_box.pth")
    parser.add_argument("--num_expressions", type=int, choices=(50, 100), default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--fov_y_deg", type=float, default=30.0)
    parser.add_argument("--camera_distance", type=float, default=1.0)
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Video not found: {args.input}")

    video_input = VideoInput(args.input)
    camera = build_camera(
        image_width=video_input.image_width,
        image_height=video_input.image_height,
        fov_y_deg=args.fov_y_deg,
        camera_distance=args.camera_distance,
        znear=args.znear,
        zfar=args.zfar,
    )

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
            output_type="torch",
        ),
        FaceBoxConfig(model_weights_path=args.facebox_weights, device=args.device),
    )
    optimizer_config = load_fitting_config(
        args.fitting_config,
        OnlineGNOptimizerConfig,
        flame=flame_config,
        device=args.device,
    )
    optimizer = OnlineGNOptimizer(optimizer_config)
    optimizer.refresh_camera_matrices(camera)

    writer: cv2.VideoWriter | None = None
    visualizer: Visualizer | None = None
    if args.output_mp4 is not None:
        args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
        output_fps = float(args.fps) if video_input.fps is None else float(video_input.fps)
        frame_size = (int(video_input.image_width * 2), int(video_input.image_height * 2))
        writer = cv2.VideoWriter(
            str(args.output_mp4),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            frame_size,
        )
        if not writer.isOpened():
            video_input.close()
            raise OSError(f"Failed to open video writer: {args.output_mp4}")
        visualizer = Visualizer(flame_assets_path=args.flame_assets, device=args.device)

    face_bbox = None
    first_frame_latency = None
    steady_state_elapsed = 0.0
    steady_state_frames = 0

    try:
        with tqdm(total=video_input.num_frames, desc="online tracking", unit="frame") as progress:
            while True:
                frame_rgb = video_input.read()
                if frame_rgb is None:
                    break

                frame_start = time.perf_counter()
                inference = inferencer.infer_image(frame_rgb, face_bbox=face_bbox)
                x = optimizer.track(inference)
                frame_elapsed = time.perf_counter() - frame_start
                face_bbox = inference.face_bbox

                if first_frame_latency is None:
                    first_frame_latency = frame_elapsed
                else:
                    steady_state_elapsed += frame_elapsed
                    steady_state_frames += 1

                if writer is not None and visualizer is not None:
                    vertex_coord = inference.vertex_coord.detach().cpu().numpy()
                    alignment = visualizer.vis_align(frame_rgb, vertex_coord)
                    alignment = visualizer.vis_contour(alignment, vertex_coord)
                    bbox = np.round(enlarge_bbox(inference.face_bbox, 1.3)).astype(np.int32)
                    cv2.rectangle(alignment, bbox[:2], bbox[2:], (255, 0, 0), 2)
                    vertices, _ = optimizer.decode(x)
                    overlay, shading = visualizer.vis_flame_shading_overlay(
                        camera,
                        frame_rgb,
                        vertices,
                        alpha=0.5,
                        return_shading=True,
                    )
                    result = np.ascontiguousarray(
                        np.concatenate(
                            [
                                np.concatenate([frame_rgb, alignment], axis=1),
                                np.concatenate([overlay, shading], axis=1),
                            ],
                            axis=0,
                        )
                    )
                    writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                progress.update(1)
    finally:
        video_input.close()
        if writer is not None:
            writer.release()
    if first_frame_latency is None:
        raise ValueError("No frames were processed.")

    print(f"First-frame latency: {first_frame_latency:.3f}s")
    if steady_state_frames > 0:
        fps = steady_state_frames / steady_state_elapsed
        print(f"Steady-state tracking: {fps:.3f} FPS ({steady_state_frames} frames)")
    if args.output_mp4 is not None:
        print(f"Saved visualization: {args.output_mp4}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
