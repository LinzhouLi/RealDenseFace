"""NeRSemble v2 multi-view FLAME tracking preprocessing.

For each selected subject and sequence, this script:
1. spatially downsamples the 16 source videos;
2. detects temporal jumps in each downsampled video;
3. runs RealDenseFace and stores one inference cache per camera;
4. fits a shared FLAME sequence from the cached multi-view observations; and
5. renders a small set of tracking visualizations.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import common.fix_chumpy  # noqa: F401  (must import before FLAME model load)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from camera import IntrinsicsCamera
from tracker import (
    FaceBoxConfig,
    FLAMEConfig,
    GNFlameOptimizer,
    GNFlameOptimizerConfig,
    InferenceOutput,
    RealDenseFaceInferenceConfig,
    RealDenseFaceInferencer,
    VideoInput,
    load_fitting_config,
)
from visualization import Visualizer, save_nersemble_tracking_visualizations


TRACKING_CAMERAS = [
    "220700191", "221501007",
    "222200036", "222200037", "222200038", "222200039", "222200040",
    "222200041", "222200042", "222200043", "222200044", "222200045",
    "222200046", "222200047", "222200048", "222200049",
]

VIS_CAMERAS = ["221501007", "222200037", "222200042"]
SKIP_SEQUENCE_NAMES = {"BACKGROUND"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RealDenseFace multi-view FLAME tracking on NeRSemble v2."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help="Comma-separated subject IDs (for example '017,018') or 'all'.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="all",
        help="Comma-separated sequence names or 'all'. BACKGROUND is always skipped.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run inference every N video frames.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Spatial video downsampling factor.",
    )
    parser.add_argument(
        "--psnr_threshold",
        type=float,
        default=20.0,
        help="Mark a frame as a temporal jump when neighbor-frame PSNR is below this value.",
    )
    parser.add_argument(
        "--visualization_frames",
        type=int,
        default=8,
        help="Number of evenly spaced fitted frames to visualize per sequence.",
    )

    parser.add_argument("--model_config", type=str, default="configs/model/vitb.yaml")
    parser.add_argument(
        "--model_weights", type=str, default="weights/realdenseface/vitb.pth"
    )
    parser.add_argument(
        "--fitting_config",
        type=str,
        default="configs/fitting/nersemble_multiview.yaml",
    )
    parser.add_argument("--flame_model", type=str, default="weights/flame/flame2023.pkl")
    parser.add_argument("--flame_assets", type=str, default="weights/flame/flame_assets.npz")
    parser.add_argument(
        "--facebox_weights", type=str, default="weights/facebox/face_box.pth"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")

    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.downsample < 1:
        parser.error("--downsample must be at least 1")
    if args.visualization_frames < 0:
        parser.error("--visualization_frames must be non-negative")
    return args


def list_subjects(dataset_root: Path, selection: str) -> list[str]:
    if selection.strip().lower() == "all":
        return [
            path.name
            for path in sorted(dataset_root.iterdir())
            if path.is_dir()
            and (path / "calibration" / "camera_params.json").is_file()
        ]
    return [item.strip() for item in selection.split(",") if item.strip()]


def list_sequences(subject_root: Path, selection: str) -> list[str]:
    sequence_root = subject_root / "sequences"
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_root}")

    available = [
        path.name
        for path in sorted(sequence_root.iterdir())
        if path.is_dir() and path.name not in SKIP_SEQUENCE_NAMES
    ]
    if selection.strip().lower() == "all":
        return available

    requested = [item.strip() for item in selection.split(",") if item.strip()]
    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(f"Sequences not found under {sequence_root}: {missing}")
    return requested


def load_camera_params(subject_root: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    calibration_path = subject_root / "calibration" / "camera_params.json"
    with calibration_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected intrinsics with shape (3, 3), got {intrinsics.shape}")

    world_to_camera = {
        serial: np.asarray(transform, dtype=np.float32)
        for serial, transform in data["world_2_cam"].items()
    }
    missing = [serial for serial in TRACKING_CAMERAS if serial not in world_to_camera]
    if missing:
        raise ValueError(f"Missing camera calibration for: {missing}")
    for serial in TRACKING_CAMERAS:
        if world_to_camera[serial].shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 world-to-camera transform for {serial}, "
                f"got {world_to_camera[serial].shape}"
            )
    return intrinsics, world_to_camera


def build_camera(
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    image_width: int,
    image_height: int,
    downsample: int,
) -> IntrinsicsCamera:
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[:2, :] /= float(downsample)
    return IntrinsicsCamera(
        K=scaled_intrinsics,
        R=world_to_camera[:3, :3],
        T=world_to_camera[:3, 3],
        width=int(image_width),
        height=int(image_height),
    )


def downsample_video(source_path: Path, output_path: Path, factor: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vf",
        f"scale=iw/{factor}:ih/{factor}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {source_path} (exit code {result.returncode}):\n"
            f"{result.stderr}"
        )


def prepare_downsample_videos(
    source_video_dir: Path,
    output_video_dir: Path,
    downsample: int,
) -> None:
    output_video_dir.mkdir(parents=True, exist_ok=True)
    for serial in tqdm(TRACKING_CAMERAS, desc="downsample videos"):
        source_path = source_video_dir / f"cam_{serial}.mp4"
        if not source_path.is_file():
            raise FileNotFoundError(f"Source video not found: {source_path}")
        downsample_video(
            source_path,
            output_video_dir / f"cam_{serial}.mp4",
            downsample,
        )


def detect_video_jump_frames(video_path: Path, psnr_threshold: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    jump_frames: list[int] = []
    previous_frame: np.ndarray | None = None
    frame_index = 0
    while True:
        valid, frame = capture.read()
        if not valid:
            break
        if previous_frame is not None:
            psnr = float(cv2.PSNR(previous_frame, frame))
            if psnr < psnr_threshold:
                jump_frames.append(frame_index)
        previous_frame = frame
        frame_index += 1
    capture.release()

    return np.asarray(jump_frames, dtype=np.int32)


def detect_and_save_jump_frames(
    video_dir: Path,
    output_path: Path,
    psnr_threshold: float,
) -> dict[str, np.ndarray]:
    jump_frames: dict[str, np.ndarray] = {}
    for serial in tqdm(TRACKING_CAMERAS, desc="detect video jumps"):
        jump_frames[serial] = detect_video_jump_frames(
            video_dir / f"cam_{serial}.mp4",
            psnr_threshold,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        camera_serials=np.asarray(TRACKING_CAMERAS),
        psnr_threshold=np.asarray(psnr_threshold, dtype=np.float32),
        **{f"cam_{serial}": indices for serial, indices in jump_frames.items()},
    )
    return jump_frames


def load_jump_frames(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            serial: data[f"cam_{serial}"].astype(np.int32, copy=False)
            for serial in TRACKING_CAMERAS
        }


def save_inference_cache(
    path: Path,
    inference: InferenceOutput,
    frame_ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_valid = (
        np.ones((len(frame_ids),), dtype=bool)
        if inference.frame_valid is None
        else np.asarray(inference.frame_valid, dtype=bool)
    )
    np.savez(
        path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        vertex_coord=np.asarray(inference.vertex_coord, dtype=np.float32),
        vertex_coord_log_var=np.asarray(inference.vertex_coord_log_var, dtype=np.float32),
        vertex_depth=np.asarray(inference.vertex_depth, dtype=np.float32),
        vertex_depth_log_var=np.asarray(inference.vertex_depth_log_var, dtype=np.float32),
        face_bbox=np.asarray(inference.face_bbox, dtype=np.float32),
        frame_valid=frame_valid,
        image_width=np.asarray(inference.image_width, dtype=np.int32),
        image_height=np.asarray(inference.image_height, dtype=np.int32),
    )


def load_inference_cache(path: Path) -> tuple[InferenceOutput, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        frame_ids = data["frame_ids"].astype(np.int32, copy=True)
        inference = InferenceOutput(
            vertex_coord=data["vertex_coord"].astype(np.float32, copy=True),
            vertex_coord_log_var=data["vertex_coord_log_var"].astype(np.float32, copy=True),
            vertex_depth=data["vertex_depth"].astype(np.float32, copy=True),
            vertex_depth_log_var=data["vertex_depth_log_var"].astype(np.float32, copy=True),
            face_bbox=data["face_bbox"].astype(np.float32, copy=True),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            frame_valid=data["frame_valid"].astype(bool, copy=True),
        )
    return inference, frame_ids


def run_sequence_inference(
    video_dir: Path,
    cache_dir: Path,
    stride: int,
    inferencer: RealDenseFaceInferencer,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for serial in tqdm(TRACKING_CAMERAS, desc="RealDenseFace inference"):
        video_input = VideoInput(video_dir / f"cam_{serial}.mp4", stride=stride)
        frame_ids = np.asarray(video_input.frame_ids, dtype=np.int32)
        inference = inferencer.infer_video_robust(video_input)
        video_input.close()
        save_inference_cache(cache_dir / f"cam_{serial}.npz", inference, frame_ids)


def fit_sequence(
    cache_dir: Path,
    jump_frames_path: Path,
    tracking_results_path: Path,
    intrinsics: np.ndarray,
    world_to_camera: dict[str, np.ndarray],
    downsample: int,
    optimizer: GNFlameOptimizer,
) -> None:
    jump_frames = load_jump_frames(jump_frames_path)
    inferences: list[InferenceOutput] = []
    cameras: list[IntrinsicsCamera] = []
    reference_frame_ids: np.ndarray | None = None
    reference_resolution: tuple[int, int] | None = None

    for serial in TRACKING_CAMERAS:
        inference, frame_ids = load_inference_cache(cache_dir / f"cam_{serial}.npz")
        if reference_frame_ids is None:
            reference_frame_ids = frame_ids
        elif not np.array_equal(reference_frame_ids, frame_ids):
            raise ValueError(f"Inference frame IDs do not match for camera {serial}")

        resolution = (int(inference.image_width), int(inference.image_height))
        if reference_resolution is None:
            reference_resolution = resolution
        elif reference_resolution != resolution:
            raise ValueError(
                f"Inference resolution does not match for camera {serial}: "
                f"{resolution} != {reference_resolution}"
            )

        mismatch = np.isin(frame_ids, jump_frames[serial])
        cached_valid = np.asarray(inference.frame_valid, dtype=bool)
        inference.frame_valid = cached_valid & ~mismatch
        inferences.append(inference)
        cameras.append(
            build_camera(
                intrinsics,
                world_to_camera[serial],
                inference.image_width,
                inference.image_height,
                downsample,
            )
        )

    if reference_frame_ids is None or reference_resolution is None:
        raise ValueError("No inference caches were loaded")

    fit_result = optimizer.fit_sequence_mv_robust(inferences, cameras)
    posed_vertices, _ = optimizer.decode(fit_result)
    num_frames = len(reference_frame_ids)

    frame_reliable = np.ones((num_frames,), dtype=bool)
    if fit_result.frame_reliable is not None:
        value = fit_result.frame_reliable
        frame_reliable = (
            value.detach().cpu().numpy().astype(bool)
            if hasattr(value, "detach")
            else np.asarray(value, dtype=bool)
        )

    frame_view_mask = np.zeros((num_frames, len(TRACKING_CAMERAS)), dtype=bool)
    if fit_result.per_frame_views is not None:
        for frame_index, views in enumerate(fit_result.per_frame_views):
            for view_index in views:
                frame_view_mask[frame_index, int(view_index)] = True

    tracking_results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        tracking_results_path,
        identity=fit_result.identity.detach().cpu().numpy().astype(np.float32),
        x=fit_result.x.detach().cpu().numpy().astype(np.float32),
        posed_vertices=posed_vertices.detach().cpu().numpy().astype(np.float32),
        frame_ids=reference_frame_ids.astype(np.int32),
        tracking_cameras=np.asarray(TRACKING_CAMERAS),
        num_expressions=np.asarray(int(fit_result.x.shape[-1]) - 18, dtype=np.int32),
        frame_reliable=frame_reliable,
        frame_view_mask=frame_view_mask,
        downsample=np.asarray(downsample, dtype=np.int32),
        image_width=np.asarray(reference_resolution[0], dtype=np.int32),
        image_height=np.asarray(reference_resolution[1], dtype=np.int32),
    )


def process_sequence(
    args: argparse.Namespace,
    subject: str,
    sequence: str,
    intrinsics: np.ndarray,
    world_to_camera: dict[str, np.ndarray],
    inferencer: RealDenseFaceInferencer,
    optimizer: GNFlameOptimizer,
    visualizer: Visualizer,
) -> None:
    subject_root = args.dataset_root / subject
    source_video_dir = subject_root / "sequences" / sequence / "images"
    tracking_dir = subject_root / "flame_tracking" / sequence
    downsample_video_dir = tracking_dir / "downsample_videos"
    inference_cache_dir = tracking_dir / "inference_cache"
    jump_frames_path = tracking_dir / "jump_frames.npz"
    tracking_results_path = tracking_dir / "tracking_results.npz"
    visualization_dir = tracking_dir / "visualization"

    print(f"[{subject}/{sequence}] downsampling videos")
    start_time = time.perf_counter()
    prepare_downsample_videos(source_video_dir, downsample_video_dir, args.downsample)
    print(f"[{subject}/{sequence}] downsampling finished in {time.perf_counter() - start_time:.1f}s")

    print(f"[{subject}/{sequence}] detecting temporal jumps")
    detect_and_save_jump_frames(
        downsample_video_dir,
        jump_frames_path,
        args.psnr_threshold,
    )

    print(f"[{subject}/{sequence}] running RealDenseFace inference")
    run_sequence_inference(
        downsample_video_dir,
        inference_cache_dir,
        args.stride,
        inferencer,
    )

    print(f"[{subject}/{sequence}] fitting multi-view FLAME sequence")
    fit_sequence(
        inference_cache_dir,
        jump_frames_path,
        tracking_results_path,
        intrinsics,
        world_to_camera,
        args.downsample,
        optimizer,
    )

    if args.visualization_frames > 0:
        print(f"[{subject}/{sequence}] rendering tracking visualizations")
        save_nersemble_tracking_visualizations(
            tracking_results_path=tracking_results_path,
            video_dir=downsample_video_dir,
            output_dir=visualization_dir,
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
            camera_serials=VIS_CAMERAS,
            visualizer=visualizer,
            downsample=args.downsample,
            num_frames=args.visualization_frames,
        )


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    flame_config = FLAMEConfig(
        flame_model_path=args.flame_model,
        flame_assets_path=args.flame_assets,
        num_expressions=100,
    )
    inferencer = RealDenseFaceInferencer(
        RealDenseFaceInferenceConfig(
            model_config_path=args.model_config,
            model_weights_path=args.model_weights,
            flame=flame_config,
            device=args.device,
            compile_model=not args.no_compile,
            filter_out_of_bounds_vertices=True,
            filter_occluded_vertices=True,
        ),
        FaceBoxConfig(model_weights_path=args.facebox_weights, device=args.device),
    )
    optimizer = GNFlameOptimizer(
        load_fitting_config(
            args.fitting_config,
            GNFlameOptimizerConfig,
            flame=flame_config,
            device=args.device,
        )
    )
    visualizer = Visualizer(flame_assets_path=args.flame_assets, device=args.device)

    subjects = list_subjects(args.dataset_root, args.subjects)
    print(f"Processing {len(subjects)} subject(s): {subjects}")
    for subject in subjects:
        subject_root = args.dataset_root / subject
        intrinsics, world_to_camera = load_camera_params(subject_root)
        sequences = list_sequences(subject_root, args.sequences)
        print(f"[{subject}] processing {len(sequences)} sequence(s): {sequences}")
        for sequence in sequences:
            process_sequence(
                args,
                subject,
                sequence,
                intrinsics,
                world_to_camera,
                inferencer,
                optimizer,
                visualizer,
            )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
