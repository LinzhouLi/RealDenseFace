from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import nvdiffrast.torch as dr

from camera import Camera


def export_mesh(
    output_path: str | Path,
    vertices: torch.Tensor,
    faces: np.ndarray,
) -> None:
    import trimesh

    vertices_np = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.export(str(output_path))


def compute_face_normal(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    i0 = faces[..., 0]
    i1 = faces[..., 1]
    i2 = faces[..., 2]
    v0 = vertices[..., i0, :]
    v1 = vertices[..., i1, :]
    v2 = vertices[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=-1)
    return face_normals


def vis_shading_mesh(
    glctx: dr.RasterizeCudaContext,
    camera: Camera,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = vertices.to(dtype=torch.float32, device="cuda")
    faces = faces.to(dtype=torch.int32, device="cuda")

    vertices_pad = torch.nn.functional.pad(vertices, [0, 1], value=1.0)
    viewproj = camera.get_full_proj.to(vertices_pad.device, torch.float32)
    vertices_clip = torch.matmul(vertices_pad, viewproj.transpose(0, 1))

    rast_out, _ = dr.rasterize(
        glctx,
        vertices_clip.unsqueeze(0),
        faces,
        resolution=(camera.height, camera.width),
    )
    rast_out = rast_out.squeeze(0)

    face_id = rast_out[..., 3:]
    mask = face_id > 0
    face_id = torch.clamp_min(face_id - 1, 0).long()

    face_normals = compute_face_normal(vertices, faces)
    normal_map = face_normals[face_id].squeeze(-2)
    shading = (normal_map * 0.5 + 0.5) * mask
    return shading, mask
