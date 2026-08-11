from __future__ import annotations

import cv2
import numpy as np


def create_default_face_bbox(image: np.ndarray, scale: float = 0.9) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    box_size = min(img_w, img_h) * float(scale)
    cx = (img_w - 1) * 0.5
    cy = (img_h - 1) * 0.5
    half = box_size * 0.5
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def enlarge_bbox(bbox: np.ndarray, scale: float) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float32)
    center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
    size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32) * float(scale)
    half = size * 0.5
    return np.array(
        [center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1]],
        dtype=np.float32,
    )


def make_bbox_square(bbox: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = np.asarray(bbox, dtype=np.float32)
    size = int(round(max(x1 - x0, y1 - y0)))
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    new_x0 = int(round(cx - size * 0.5))
    new_y0 = int(round(cy - size * 0.5))
    new_x1 = new_x0 + size
    new_y1 = new_y0 + size
    return np.array([new_x0, new_y0, new_x1, new_y1], dtype=np.int32)


def calc_crop_pad_info(
    bbox: np.ndarray,
    img_width: int,
    img_height: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    x0, y0, x1, y1 = np.asarray(bbox, dtype=np.int32)
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - img_width)
    pad_bottom = max(0, y1 - img_height)
    crop_left = max(0, x0)
    crop_top = max(0, y0)
    crop_right = min(img_width, x1)
    crop_bottom = min(img_height, y1)
    return (pad_left, pad_top, pad_right, pad_bottom), (crop_left, crop_top, crop_right, crop_bottom)


def crop_and_resize_image(
    image: np.ndarray,
    bbox: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    img_h, img_w = image.shape[:2]
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    pad_info, crop_info = calc_crop_pad_info(bbox, img_w, img_h)
    bbox_w = int(round(float(bbox[2] - bbox[0])))
    bbox_h = int(round(float(bbox[3] - bbox[1])))

    x0, y0, x1, y1 = crop_info
    cropped = image[y0:y1, x0:x1]
    if sum(pad_info) > 0:
        padded = np.zeros((bbox_h, bbox_w, 3), dtype=image.dtype)
        padded[pad_info[1]:(bbox_h - pad_info[3]), pad_info[0]:(bbox_w - pad_info[2])] = cropped
        cropped = padded

    resized = cv2.resize(cropped, (int(target_size), int(target_size)), interpolation=cv2.INTER_LINEAR)
    return resized, bbox
