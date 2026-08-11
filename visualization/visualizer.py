from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from camera import Camera
from .mesh import compute_face_normal, vis_shading_mesh


iris_vertex_array = np.array([
    3931,3932,3933,3935,3936,3937,3939,3940,3941,3943,3944,3945,3947,3948,
    3949,3951,3952,3953,3955,3956,3957,3959,3960,3961,3963,3964,3965,3967,
    3968,3969,3971,3972,3973,3975,3976,3977,3979,3980,3981,3983,3984,3985,
    3987,3988,3989,3991,3992,3993,3995,3996,3997,3999,4000,4001,4003,4004,
    4005,4007,4008,4009,4011,4012,4013,4015,4016,4017,4019,4020,4021,4023,
    4024,4025,4027,4028,4029,4031,4032,4033,4035,4036,4037,4039,4040,4041,
    4043,4044,4045,4047,4048,4049,4051,4052,4053,4054,4056,4057,4058,4477,
    4478,4479,4481,4482,4483,4485,4486,4487,4489,4490,4491,4493,4494,4495,
    4497,4498,4499,4501,4502,4503,4505,4506,4507,4509,4510,4511,4513,4514,
    4515,4517,4518,4519,4521,4522,4523,4525,4526,4527,4529,4530,4531,4533,
    4534,4535,4537,4538,4539,4541,4542,4543,4545,4546,4547,4549,4550,4551,
    4553,4554,4555,4557,4558,4559,4561,4562,4563,4565,4566,4567,4569,4570,
    4571,4573,4574,4575,4577,4578,4579,4581,4582,4583,4585,4586,4587,4589,
    4590,4591,4593,4594,4595,4597,4598,4599,4600,4602,4603,4604
], dtype=np.int32)


class Visualizer:
    def __init__(
        self,
        flame_assets_path: str = "weights/flame/flame_assets.npz",
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.glctx = dr.RasterizeCudaContext()

        flame_assets = np.load(flame_assets_path)
        self.vis_edges = np.asarray(flame_assets["vis_edges"], dtype=np.int32)
        faces_t = torch.from_numpy(np.asarray(flame_assets["faces"]))
        self.faces_int32 = faces_t.to(dtype=torch.int32, device=self.device)
        self.faces_int64 = faces_t.to(dtype=torch.int64, device=self.device)

        num_vertices = int(self.faces_int64.max().item()) + 1
        base_vertex_colors = torch.ones((num_vertices, 3), dtype=torch.float32, device=self.device)
        iris_idx = torch.from_numpy(iris_vertex_array).to(dtype=torch.long, device=self.device)
        base_vertex_colors[iris_idx] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        self.base_vertex_colors = base_vertex_colors.contiguous()

    def vis_align(self, image: np.ndarray, vertex_coord: np.ndarray, line_width: int = 1) -> np.ndarray:
        image_vis = image.copy()
        vertex_xy = np.asarray(vertex_coord, dtype=np.int32)
        for edge in self.vis_edges:
            x0, y0, x1, y1 = vertex_xy[edge].reshape(-1).tolist()
            cv2.line(image_vis, (x0, y0), (x1, y1), (0, 127, 255), line_width, lineType=cv2.LINE_AA)
        return image_vis

    def vis_contour(self, image: np.ndarray, vertex_coord: np.ndarray, line_width: int = 1) -> np.ndarray:
        image_vis = image.copy()
        h, w = image_vis.shape[:2]

        vertices = torch.from_numpy(np.asarray(vertex_coord, dtype=np.float32)).to(self.device)
        vertices_ndc = torch.zeros((vertices.shape[0], 4), dtype=torch.float32, device=self.device)
        vertices_ndc[:, 0] = vertices[:, 0] / float(w) * 2.0 - 1.0
        vertices_ndc[:, 1] = vertices[:, 1] / float(h) * 2.0 - 1.0
        vertices_ndc[:, 2] = 0.5
        vertices_ndc[:, 3] = 1.0

        rast_out = dr.rasterize(
            self.glctx,
            vertices_ndc.unsqueeze(0),
            self.faces_int32,
            resolution=[h, w],
        )[0].squeeze(0)
        mask = (rast_out[..., 3:] > 0).to(dtype=torch.uint8)
        mask = (mask * 255).cpu().numpy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(image_vis, [cnt], -1, (0, 255, 0), line_width)
        return image_vis

    def vis_flame_normal_overlay(
        self,
        camera: Camera,
        image: np.ndarray,
        vertices: np.ndarray | torch.Tensor,
        alpha: float = 0.5,
        return_normal_map: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        image_tf = torch.from_numpy(np.asarray(image, dtype=np.uint8)).to(dtype=torch.float32, device=self.device) / 255.0
        if isinstance(vertices, np.ndarray):
            vertices_tf = torch.from_numpy(vertices).to(dtype=torch.float32, device=self.device)
        else:
            vertices_tf = vertices.to(dtype=torch.float32, device=self.device)

        normal_map, mask = vis_shading_mesh(self.glctx, camera, vertices_tf, self.faces_int64)
        overlay = (image_tf * (1.0 - alpha) + normal_map * alpha) * mask + image_tf * (~mask)
        normal_map = (normal_map * 255.0).to(dtype=torch.uint8, device="cpu").numpy()
        overlay = (overlay * 255.0).to(dtype=torch.uint8, device="cpu").numpy()
        overlay = np.ascontiguousarray(overlay)
        normal_map = np.ascontiguousarray(normal_map)
        if return_normal_map:
            return overlay, normal_map
        return overlay

    def vis_flame_shading_overlay(
        self,
        camera: Camera,
        image: np.ndarray,
        vertices: np.ndarray | torch.Tensor,
        alpha: float = 0.5,
        return_shading: bool = False,
        shading_background: str = "black",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        image_tf = torch.from_numpy(np.asarray(image, dtype=np.uint8)).to(dtype=torch.float32, device=self.device) / 255.0
        if isinstance(vertices, np.ndarray):
            vertices_tf = torch.from_numpy(vertices).to(dtype=torch.float32, device=self.device)
        else:
            vertices_tf = vertices.to(dtype=torch.float32, device=self.device)

        vertices_pad = F.pad(vertices_tf, [0, 1], value=1.0)
        viewproj = camera.get_full_proj.to(device=self.device, dtype=torch.float32)
        vertices_clip = torch.matmul(vertices_pad, viewproj.transpose(0, 1))
        rast_out, _ = dr.rasterize(
            self.glctx,
            vertices_clip.unsqueeze(0),
            self.faces_int32,
            resolution=(camera.height, camera.width),
        )
        rast_squeezed = rast_out.squeeze(0)

        face_id = rast_squeezed[..., 3:]
        mask = face_id > 0
        face_id = torch.clamp_min(face_id - 1, 0).long()

        face_normals = compute_face_normal(vertices_tf, self.faces_int64)
        normal_map_world = face_normals[face_id].squeeze(-2)
        view_rot = camera.get_w2v[:3, :3].to(device=self.device, dtype=torch.float32)
        normal_map_view = torch.matmul(normal_map_world, view_rot.transpose(0, 1))
        normal_map_view = F.normalize(normal_map_view, dim=-1) * -1.0

        light_dir = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32, device=self.device)
        light_dir = F.normalize(light_dir, dim=0).view(1, 1, 3)
        diffuse = torch.clamp_min(torch.sum(normal_map_view * light_dir, dim=-1, keepdim=True), 0.0)
        shading_scalar = torch.clamp(0.15 + 0.85 * diffuse, 0.0, 1.0)

        color_map, _ = dr.interpolate(self.base_vertex_colors, rast_out, self.faces_int32)
        color_map = color_map.squeeze(0)
        shading = color_map * shading_scalar

        overlay = (image_tf * (1.0 - alpha) + shading * alpha) * mask + image_tf * (~mask)
        if shading_background == "black":
            shading = shading * mask
        elif shading_background == "white":
            shading = shading * mask + (~mask).to(dtype=shading.dtype)
        else:
            raise ValueError("shading_background must be 'black' or 'white'")
        shading = (shading * 255.0).to(dtype=torch.uint8, device="cpu").numpy()
        overlay = (overlay * 255.0).to(dtype=torch.uint8, device="cpu").numpy()
        overlay = np.ascontiguousarray(overlay)
        shading = np.ascontiguousarray(shading)
        if return_shading:
            return overlay, shading
        return overlay

