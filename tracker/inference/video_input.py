from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VideoInput:
    def __init__(
        self,
        source: str | Path,
        *,
        K: np.ndarray | None = None,
        D: np.ndarray | None = None,
        stride: int = 1,
    ) -> None:
        self.source = Path(source)
        self._mode = ""
        self._cap: cv2.VideoCapture | None = None
        self._frame_paths: list[Path] = []
        # `_sample_index` counts frames returned by read() (after stride subsampling).
        # `_cap_pos` tracks the next raw frame index the video capture will return
        # (grab-only increments this without decoding, so we can cheaply skip).
        self._sample_index = 0
        self._cap_pos = 0
        self._total_frames = 0
        self._num_frames = 0
        self._image_width = 0
        self._image_height = 0
        self._fps: float | None = None
        self._stride = max(1, int(stride))

        # Optional OpenCV pinhole intrinsics + distortion. When both are provided
        # (and D has any non-zero element), every frame returned by read() /
        # read_frame() is automatically passed through cv2.undistort. This keeps
        # downstream inference / camera projection in a clean pinhole space and
        # avoids the caller having to wrap VideoInput just for undistortion.
        self._K = None if K is None else np.asarray(K, dtype=np.float32)
        self._D = None if D is None else np.asarray(D, dtype=np.float32).reshape(-1)
        self._undistort_enabled = (
            self._K is not None
            and self._D is not None
            and bool(np.any(self._D != 0))
        )

        if self.source.is_dir():
            self._mode = "frames"
            self._frame_paths = [
                path for path in self.source.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            ]
            self._frame_paths.sort()
            if len(self._frame_paths) == 0:
                raise ValueError(f"No image frames found in directory: {self.source}")
            first_frame = cv2.imread(str(self._frame_paths[0]), cv2.IMREAD_COLOR)
            if first_frame is None:
                raise ValueError(f"Failed to read first frame: {self._frame_paths[0]}")
            self._total_frames = len(self._frame_paths)
            self._image_height, self._image_width = first_frame.shape[:2]
        elif self.source.is_file():
            self._mode = "video"
            self._cap = cv2.VideoCapture(str(self.source))
            if not self._cap.isOpened():
                raise ValueError(f"Failed to open video: {self.source}")
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._image_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._image_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            self._fps = fps if fps > 0 else None
        else:
            raise ValueError(f"VideoInput source does not exist: {self.source}")

        # Sampled frame indices (into the raw video) after stride subsampling.
        # With stride == 1 this is [0, 1, ..., total_frames - 1] and behavior is
        # backwards-compatible.
        self._frame_ids = np.arange(0, self._total_frames, self._stride, dtype=np.int64)
        self._num_frames = int(self._frame_ids.size)

    @property
    def num_frames(self) -> int:
        """Number of frames yielded by read() (after stride subsampling)."""
        return self._num_frames

    @property
    def total_num_frames(self) -> int:
        """Raw frame count of the underlying video / directory, ignoring stride."""
        return self._total_frames

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def frame_ids(self) -> np.ndarray:
        """Raw frame indices corresponding to each read() call, shape (num_frames,)."""
        return self._frame_ids

    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @property
    def fps(self) -> float | None:
        return self._fps

    def reset(self) -> None:
        self._sample_index = 0
        self._cap_pos = 0
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _maybe_undistort(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._undistort_enabled:
            return cv2.undistort(frame_bgr, self._K, self._D)
        return frame_bgr

    def read_frame(self, frame_index: int) -> np.ndarray:
        """Random-access read by *raw* frame index (unaffected by stride)."""
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= self._total_frames:
            raise IndexError(f"frame_index out of range: {frame_index}")

        if self._mode == "frames":
            frame_path = self._frame_paths[frame_index]
            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError(f"Failed to read frame: {frame_path}")
            frame_bgr = self._maybe_undistort(frame_bgr)
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._cap is None:
            raise ValueError("VideoInput is not initialized with a readable source.")

        saved_pos = self._cap_pos
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame_bgr = self._cap.read()
        # Restore the streaming pointer so an interleaved read() sequence keeps its place.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
        if not ret:
            raise ValueError(f"Failed to read frame at index: {frame_index}")
        frame_bgr = self._maybe_undistort(frame_bgr)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def read(self) -> np.ndarray | None:
        """Return the next sampled frame (advances by `stride` under the hood)."""
        if self._sample_index >= self._num_frames:
            return None
        target = int(self._frame_ids[self._sample_index])
        self._sample_index += 1

        if self._mode == "frames":
            frame_path = self._frame_paths[target]
            frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError(f"Failed to read frame: {frame_path}")
            frame_bgr = self._maybe_undistort(frame_bgr)
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._cap is None:
            raise ValueError("VideoInput is not initialized with a readable source.")

        # Cheap forward skip via grab() (does not decode), then decode the target frame.
        # For stride == 1 this reduces to the plain read() path (single grab+retrieve).
        while self._cap_pos < target:
            if not self._cap.grab():
                return None
            self._cap_pos += 1
        ret, frame_bgr = self._cap.read()
        if not ret:
            return None
        self._cap_pos += 1
        frame_bgr = self._maybe_undistort(frame_bgr)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
