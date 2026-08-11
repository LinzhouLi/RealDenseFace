from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from camera import Camera
from tracker.inference.preprocess import enlarge_bbox

from .visualizer import Visualizer


def save_tracking_video(
    output_path: str | Path,
    video_input,
    inference,
    camera: Camera,
    visualizer: Visualizer,
    vertices: torch.Tensor,
    fallback_fps: float,
) -> None:
    fps = float(fallback_fps) if video_input.fps is None else float(video_input.fps)
    video_input.reset()
    frame_size = (int(inference.image_width * 2), int(inference.image_height * 2))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
    )
    if not writer.isOpened():
        raise OSError(f"Failed to open video writer: {output_path}")

    try:
        num_frames = int(inference.vertex_coord.shape[0])
        for frame_id in tqdm(range(num_frames), desc="visualize video"):
            frame_rgb = video_input.read()
            if frame_rgb is None:
                raise ValueError(f"Unexpected end of input at frame {frame_id}.")

            alignment = visualizer.vis_align(frame_rgb, inference.vertex_coord[frame_id])
            alignment = visualizer.vis_contour(alignment, inference.vertex_coord[frame_id])
            bbox = np.round(enlarge_bbox(inference.face_bbox[frame_id], 1.3)).astype(np.int32)
            cv2.rectangle(alignment, bbox[:2], bbox[2:], (255, 0, 0), 2)
            overlay, shading = visualizer.vis_flame_shading_overlay(
                camera, frame_rgb, vertices[frame_id], alpha=0.5, return_shading=True
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
    finally:
        writer.release()
