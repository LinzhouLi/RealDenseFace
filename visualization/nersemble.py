from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from camera import IntrinsicsCamera

from .visualizer import Visualizer


def save_nersemble_tracking_visualizations(
    tracking_results_path: Path,
    video_dir: Path,
    output_dir: Path,
    intrinsics: np.ndarray,
    world_to_camera: dict[str, np.ndarray],
    camera_serials: list[str],
    visualizer: Visualizer,
    downsample: int,
    num_frames: int = 8,
) -> None:
    with np.load(tracking_results_path, allow_pickle=False) as flame_data:
        frame_ids = flame_data["frame_ids"].astype(np.int32, copy=False)
        posed_vertices = flame_data["posed_vertices"].astype(np.float32, copy=False)
        width = int(flame_data["image_width"])
        height = int(flame_data["image_height"])
    if len(frame_ids) == 0:
        return

    selected = np.linspace(0, len(frame_ids) - 1, min(num_frames, len(frame_ids)), dtype=np.int32)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_image in output_dir.glob("frame_*.jpg"):
        stale_image.unlink()

    captures: dict[str, cv2.VideoCapture] = {}
    cameras: dict[str, IntrinsicsCamera] = {}
    scaled_intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    scaled_intrinsics[:2, :] /= float(downsample)
    for serial in camera_serials:
        video_path = video_dir / f"cam_{serial}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            for opened_capture in captures.values():
                opened_capture.release()
            raise RuntimeError(f"Failed to open visualization video: {video_path}")
        captures[serial] = capture
        transform = world_to_camera[serial]
        cameras[serial] = IntrinsicsCamera(
            K=scaled_intrinsics,
            R=transform[:3, :3],
            T=transform[:3, 3],
            width=width,
            height=height,
        )

    try:
        for sample_index in selected.tolist():
            frame_id = int(frame_ids[sample_index])
            panels = []
            for serial in camera_serials:
                capture = captures[serial]
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                valid, frame_bgr = capture.read()
                if not valid:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                panels.append(
                    visualizer.vis_flame_shading_overlay(
                        cameras[serial],
                        frame_rgb,
                        posed_vertices[sample_index],
                        alpha=0.5,
                    )
                )
            if panels:
                composite = np.concatenate(panels, axis=1)
                output_path = output_dir / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
    finally:
        for capture in captures.values():
            capture.release()
