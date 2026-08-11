from __future__ import annotations

import cv2
import nvdiffrast.torch as dr
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.facebox import FaceDetector
from models.realdenseface import RealDenseFace
from common.config import (
    INVISIBLE_LOG_VAR,
    FaceBoxConfig,
    RealDenseFaceInferenceConfig,
    load_model_config,
)
from common.types import InferenceOutput

from .preprocess import create_default_face_bbox, crop_and_resize_image, enlarge_bbox, make_bbox_square
from .video_input import VideoInput


NUM_FLAME_VERTICES = 5023


class RealDenseFaceInferencer:
    def __init__(
        self,
        config: RealDenseFaceInferenceConfig,
        facebox_config: FaceBoxConfig | None = None,
    ) -> None:
        self.config = config
        device_name = config.device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)

        self.model_config = load_model_config(config.model_config_path)
        self.model_config["dino_pretrained"] = False
        self.target_size = int(self.model_config["target_size"])
        self.scale_coord = self.model_config.get("scale_coord", 1.0)
        self.scale_depth = self.model_config.get("scale_depth", 1.0)

        self.model = RealDenseFace(**self.model_config)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(config.model_weights_path, map_location=self.device))
        self.model.eval()
        self.model.build_inference_cache()

        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        flame_assets = np.load(config.flame.flame_assets_path)
        self.vis_edges = flame_assets["vis_edges"]
        self.flame_vertex_uv = torch.from_numpy(flame_assets["vertex_uvs"]).to(self.device).unsqueeze(0).unsqueeze(2) * 2.0 - 1.0
        self.key_vertex_ids = torch.from_numpy(self.vis_edges).to(dtype=torch.int64, device=self.device).flatten().unique()
        self.faces_int32 = torch.from_numpy(np.asarray(flame_assets["faces"], dtype=np.int32)).to(dtype=torch.int32, device=self.device).contiguous()
        self.visibility_glctx = dr.RasterizeCudaContext()

        self.facebox_config = FaceBoxConfig() if facebox_config is None else facebox_config
        self.face_detector = self._create_face_detector()

        self._input_tensor = torch.empty((self.target_size, self.target_size, 3), dtype=torch.uint8)
        if self.device.type == "cuda":
            self._input_tensor = self._input_tensor.pin_memory()
        self._input_array = self._input_tensor.numpy()

        if config.compile_model and hasattr(torch, "compile"):
            self._compile()

    def _create_face_detector(self) -> FaceDetector:
        config = self.facebox_config
        return FaceDetector(
            model_path=config.model_weights_path,
            image_size=(int(config.input_height), int(config.input_width)),
            threshold=float(config.score_threshold),
            half=bool(config.half),
            device=config.device,
        )

    def _compile(self) -> None:
        dummy = torch.zeros((self.target_size, self.target_size, 3), dtype=torch.uint8, device=self.device)
        for _ in range(3): self._run(dummy)
        self._run = torch.compile(self._run, mode="max-autotune", dynamic=False)
        for _ in range(3): self._run(dummy)

    @torch.no_grad()
    def _run(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = image.permute(2, 0, 1).unsqueeze(0).contiguous()
        image = image.to(torch.float32) / 255.0
        image = (image - self.image_mean) / self.image_std

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            coords, coord_log_vars, depths, depth_log_vars = self.model(image)
            coords = coords / self.scale_coord
            depths = depths / self.scale_depth

        coords = coords * 0.5 + 0.5
        coord_log_vars = torch.clamp(coord_log_vars, min=-8.0, max=3.0)
        depth_log_vars = torch.clamp(depth_log_vars, min=-8.0, max=3.0)

        vertex_coord = F.grid_sample(coords.float(), self.flame_vertex_uv, mode="bilinear", align_corners=False).squeeze(-1)
        vertex_coord_log_var = F.grid_sample(coord_log_vars.float(), self.flame_vertex_uv, mode="bilinear", align_corners=False).squeeze(-1)
        vertex_depth = F.grid_sample(depths.float(), self.flame_vertex_uv, mode="bilinear", align_corners=False).squeeze(-1)
        vertex_depth_log_var = F.grid_sample(depth_log_vars.float(), self.flame_vertex_uv, mode="bilinear", align_corners=False).squeeze(-1)

        vertex_coord = vertex_coord.squeeze(0).permute(1, 0).contiguous()
        vertex_coord_log_var = vertex_coord_log_var.view(5023, 1)
        vertex_depth = vertex_depth.view(5023, 1)
        vertex_depth_log_var = vertex_depth_log_var.view(5023, 1)
        return vertex_coord, vertex_coord_log_var, vertex_depth, vertex_depth_log_var

    @torch.inference_mode()
    def _detect_face_bbox(self, image: np.ndarray) -> np.ndarray | None:
        target_h = int(self.facebox_config.input_height)
        target_w = int(self.facebox_config.input_width)
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).to(self.face_detector.device, dtype=torch.float32) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        box, score = self.face_detector.detect_one(tensor)
        if score < float(self.facebox_config.score_threshold):
            return None

        box = np.asarray(box, dtype=np.float32)
        scale_x = image.shape[1] / float(target_w)
        scale_y = image.shape[0] / float(target_h)
        box[0::2] *= scale_x
        box[1::2] *= scale_y
        return box

    def _init_face_bbox(self, image: np.ndarray) -> np.ndarray:
        detected = self._detect_face_bbox(image)
        if detected is None:
            detected = create_default_face_bbox(image)
        output = self.run(image, detected) # It is necessary to refine bbox by our own model
        return output.face_bbox

    @torch.inference_mode()
    def run(self, image: np.ndarray, face_bbox: np.ndarray) -> InferenceOutput:
        img_h, img_w = image.shape[:2]

        with torch.profiler.record_function("realdenseface_run_bbox_sanitize"):
            face_bbox = np.asarray(face_bbox, dtype=np.float32).copy()
            face_bbox[0] = max(face_bbox[0], 0.0)
            face_bbox[1] = max(face_bbox[1], 0.0)
            face_bbox[2] = min(face_bbox[2], float(img_w))
            face_bbox[3] = min(face_bbox[3], float(img_h))

        with torch.profiler.record_function("realdenseface_run_bbox_transform"):
            face_bbox = enlarge_bbox(face_bbox, self.config.enlarge_bbox_ratio)
            face_bbox = make_bbox_square(face_bbox)

        with torch.profiler.record_function("realdenseface_run_crop_resize"):
            image_crop, face_bbox = crop_and_resize_image(image, face_bbox, self.target_size)

        with torch.profiler.record_function("realdenseface_run_h2d"):
            np.copyto(self._input_array, image_crop)
            image_tensor = self._input_tensor.to(self.device, non_blocking=True)

        with torch.profiler.record_function("realdenseface_run_core"):
            vertex_coord, vertex_coord_log_var, vertex_depth, vertex_depth_log_var = self._run(image_tensor)

            if bool(self.config.filter_occluded_vertices):
                vertex_coord_log_var, vertex_depth_log_var = self.filter_visibility(
                    vertex_coord,
                    vertex_depth,
                    vertex_coord_log_var,
                    vertex_depth_log_var,
                )

        with torch.profiler.record_function("realdenseface_run_remap_coords"):
            bbox_w = float(face_bbox[2] - face_bbox[0])
            bbox_h = float(face_bbox[3] - face_bbox[1])
            vertex_coord[:, 0] = vertex_coord[:, 0] * bbox_w + face_bbox[0]
            vertex_coord[:, 1] = vertex_coord[:, 1] * bbox_h + face_bbox[1]

            if bool(self.config.filter_out_of_bounds_vertices):
                vertex_coord_log_var, vertex_depth_log_var = self.filter_image_bounds(
                    vertex_coord,
                    vertex_coord_log_var,
                    vertex_depth_log_var,
                    img_w,
                    img_h,
                )

        with torch.profiler.record_function("realdenseface_run_update_bbox"):
            key_vertex_coord = vertex_coord[self.key_vertex_ids]
            coord_min = torch.min(key_vertex_coord, dim=0).values
            coord_max = torch.max(key_vertex_coord, dim=0).values
            new_bbox = torch.stack([coord_min[0], coord_min[1], coord_max[0], coord_max[1]])
            new_bbox = new_bbox.cpu().numpy().astype(np.float32, copy=False)

        if self.config.output_type == "numpy":
            with torch.profiler.record_function("realdenseface_run_pack_output_numpy"):
                return InferenceOutput(
                    vertex_coord=vertex_coord.detach().cpu().numpy().astype(np.float32, copy=False),
                    vertex_coord_log_var=vertex_coord_log_var.detach().cpu().numpy().astype(np.float32, copy=False),
                    vertex_depth=vertex_depth.detach().cpu().numpy().astype(np.float32, copy=False),
                    vertex_depth_log_var=vertex_depth_log_var.detach().cpu().numpy().astype(np.float32, copy=False),
                    face_bbox=new_bbox,
                    image_width=img_w,
                    image_height=img_h,
                )
        elif self.config.output_type == "torch":
            with torch.profiler.record_function("realdenseface_run_pack_output_torch"):
                return InferenceOutput(
                    vertex_coord=vertex_coord.to(torch.float32),
                    vertex_coord_log_var=vertex_coord_log_var.to(torch.float32),
                    vertex_depth=vertex_depth.to(torch.float32),
                    vertex_depth_log_var=vertex_depth_log_var.to(torch.float32),
                    face_bbox=new_bbox,
                    image_width=img_w,
                    image_height=img_h,
                )
        else:
            raise NotImplementedError


    @torch.inference_mode()
    def filter_visibility(
        self,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        vertex_depth_log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible = self._compute_raster_visibility(vertex_coord, vertex_depth[:, 0])
        vertex_coord_log_var = vertex_coord_log_var.clone()
        vertex_depth_log_var = vertex_depth_log_var.clone()
        vertex_coord_log_var[~visible] = float(self.config.filter_invisible_log_var)
        vertex_depth_log_var[~visible] = float(self.config.filter_invisible_log_var)
        return vertex_coord_log_var, vertex_depth_log_var

    @torch.inference_mode()
    def filter_image_bounds(
        self,
        vertex_coord: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        vertex_depth_log_var: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visible = (
            (vertex_coord[:, 0] >= 0.0)
            & (vertex_coord[:, 0] <= float(image_width))
            & (vertex_coord[:, 1] >= 0.0)
            & (vertex_coord[:, 1] <= float(image_height))
        )
        vertex_coord_log_var = vertex_coord_log_var.clone()
        vertex_depth_log_var = vertex_depth_log_var.clone()
        vertex_coord_log_var[~visible] = float(self.config.filter_invisible_log_var)
        vertex_depth_log_var[~visible] = float(self.config.filter_invisible_log_var)
        return vertex_coord_log_var, vertex_depth_log_var

    def _compute_raster_visibility(
        self,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
    ) -> torch.Tensor:
        depth_min = torch.min(vertex_depth)
        depth_max = torch.max(vertex_depth)
        depth_order = (vertex_depth - depth_min) / (depth_max - depth_min)

        vertices_clip = torch.empty((NUM_FLAME_VERTICES, 4), dtype=torch.float32, device=self.device)
        vertices_clip[:, 0] = vertex_coord[:, 0] * 2.0 - 1.0
        vertices_clip[:, 1] = vertex_coord[:, 1] * 2.0 - 1.0
        vertices_clip[:, 2] = depth_order
        vertices_clip[:, 3] = 1.0

        rast_out, _ = dr.rasterize(
            self.visibility_glctx,
            vertices_clip.unsqueeze(0).contiguous(),
            self.faces_int32,
            resolution=(self.target_size, self.target_size),
        )
        rast_out = rast_out.squeeze(0)
        rast_mask = rast_out[..., 3] > 0
        rast_depth = (rast_out[..., 2] + 1.0) * rast_mask.to(dtype=torch.float32)
        rast_depth_image = rast_depth.view(1, 1, self.target_size, self.target_size)

        # contour protection
        rast_depth_dilated = F.max_pool2d(rast_depth_image, kernel_size=3, stride=1, padding=1)
        rast_depth_for_sample = torch.where(
            rast_mask.view(1, 1, self.target_size, self.target_size),
            rast_depth_image,
            rast_depth_dilated,
        )

        sample_grid = (vertex_coord * 2.0 - 1.0).view(1, NUM_FLAME_VERTICES, 1, 2)
        sample_depth = F.grid_sample(
            rast_depth_for_sample,
            sample_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).view(NUM_FLAME_VERTICES)
        return torch.abs(sample_depth - (depth_order + 1.0)) <= float(self.config.filter_visibility_depth_eps)

    @torch.inference_mode()
    def infer_image(self, image: np.ndarray, face_bbox: np.ndarray | None = None) -> InferenceOutput:
        if face_bbox is None:
            face_bbox = self._init_face_bbox(image)
        return self.run(image, face_bbox)

    @torch.inference_mode()
    def infer_video(
        self,
        video_input: VideoInput,
        init_face_bbox: np.ndarray | None = None,
        discontinuous: bool = False,
    ) -> InferenceOutput:
        video_input.reset()

        num_frames = int(video_input.num_frames)
        image_width = int(video_input.image_width)
        image_height = int(video_input.image_height)
        if num_frames <= 0:
            raise ValueError("infer_video requires a valid video frame count.")

        vertex_coords = np.empty((num_frames, NUM_FLAME_VERTICES, 2), dtype=np.float32)
        vertex_coord_log_vars = np.empty((num_frames, NUM_FLAME_VERTICES, 1), dtype=np.float32)
        vertex_depths = np.empty((num_frames, NUM_FLAME_VERTICES, 1), dtype=np.float32)
        vertex_depth_log_vars = np.empty((num_frames, NUM_FLAME_VERTICES, 1), dtype=np.float32)
        face_bboxes = np.empty((num_frames, 4), dtype=np.float32)

        face_bbox = None if discontinuous else init_face_bbox
        valid_frames = 0
        for _ in tqdm(range(num_frames), desc="infer video"):
            frame_rgb = video_input.read()
            if frame_rgb is None:
                break

            if face_bbox is None:
                face_bbox = self._init_face_bbox(frame_rgb)

            output = self.run(frame_rgb, face_bbox)
            face_bbox = None if discontinuous else output.face_bbox

            vertex_coords[valid_frames] = output.vertex_coord
            vertex_coord_log_vars[valid_frames] = output.vertex_coord_log_var
            vertex_depths[valid_frames] = output.vertex_depth
            vertex_depth_log_vars[valid_frames] = output.vertex_depth_log_var
            face_bboxes[valid_frames] = output.face_bbox
            valid_frames += 1

        if valid_frames == 0:
            raise ValueError("infer_video received an empty video stream.")

        return InferenceOutput(
            vertex_coord=vertex_coords[:valid_frames],
            vertex_coord_log_var=vertex_coord_log_vars[:valid_frames],
            vertex_depth=vertex_depths[:valid_frames],
            vertex_depth_log_var=vertex_depth_log_vars[:valid_frames],
            face_bbox=face_bboxes[:valid_frames],
            image_width=image_width,
            image_height=image_height,
        )

    @torch.inference_mode()
    def infer_video_robust(self, video_input: VideoInput) -> InferenceOutput:
        video_input.reset()

        num_frames = int(video_input.num_frames)
        image_width = int(video_input.image_width)
        image_height = int(video_input.image_height)
        if num_frames <= 0:
            raise ValueError("infer_video_robust requires a valid video frame count.")

        invisible_log_var = INVISIBLE_LOG_VAR

        vertex_coords = np.zeros((num_frames, NUM_FLAME_VERTICES, 2), dtype=np.float32)
        vertex_coord_log_vars = np.full((num_frames, NUM_FLAME_VERTICES, 1), invisible_log_var, dtype=np.float32)
        vertex_depths = np.zeros((num_frames, NUM_FLAME_VERTICES, 1), dtype=np.float32)
        vertex_depth_log_vars = np.full((num_frames, NUM_FLAME_VERTICES, 1), invisible_log_var, dtype=np.float32)
        face_bboxes = np.zeros((num_frames, 4), dtype=np.float32)
        frame_valid = np.zeros((num_frames,), dtype=bool)

        prev_bbox: np.ndarray | None = None

        for t in tqdm(range(num_frames), desc="infer video (robust)"):
            frame_rgb = video_input.read()
            if frame_rgb is None:
                break

            detected = self._detect_face_bbox(frame_rgb)

            if detected is None:
                prev_bbox = None
                continue

            if prev_bbox is None:
                output = self.run(frame_rgb, detected)
                output = self.run(frame_rgb, output.face_bbox)
            else:
                output = self.run(frame_rgb, prev_bbox)

            prev_bbox = output.face_bbox
            frame_valid[t] = True
            vertex_coords[t] = output.vertex_coord
            vertex_coord_log_vars[t] = output.vertex_coord_log_var
            vertex_depths[t] = output.vertex_depth
            vertex_depth_log_vars[t] = output.vertex_depth_log_var
            face_bboxes[t] = output.face_bbox

        return InferenceOutput(
            vertex_coord=vertex_coords,
            vertex_coord_log_var=vertex_coord_log_vars,
            vertex_depth=vertex_depths,
            vertex_depth_log_var=vertex_depth_log_vars,
            face_bbox=face_bboxes,
            image_width=image_width,
            image_height=image_height,
            frame_valid=frame_valid,
        )

