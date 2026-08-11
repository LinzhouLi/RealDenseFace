from __future__ import annotations

import math

import numpy as np
import torch

from camera import Camera, PerspectiveCamera


def _single_camera_matrices(optimizer, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    view_mat = camera.get_w2v.to(optimizer.device, torch.float32).unsqueeze(0)
    viewproj_mat = camera.get_full_proj.to(optimizer.device, torch.float32).unsqueeze(0)
    return view_mat, viewproj_mat


def _build_candidate_camera(base_camera: Camera, fov_y_degrees: float) -> PerspectiveCamera:
    rot = base_camera.get_v2w[:3, :3].detach().cpu().numpy().astype(np.float32, copy=False)
    pos = base_camera.get_pos.detach().cpu().numpy().astype(np.float32, copy=False)
    return PerspectiveCamera(
        fov_y=math.radians(float(fov_y_degrees)),
        rot=rot,
        pos=pos,
        width=int(base_camera.width),
        height=int(base_camera.height),
        znear=float(base_camera.znear),
        zfar=float(base_camera.zfar),
    )


def _normalize_single_frame_observations(
    optimizer,
    inference,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertex_coord = torch.as_tensor(inference.vertex_coord, dtype=torch.float32, device=optimizer.device).clone()
    vertex_depth = torch.as_tensor(inference.vertex_depth, dtype=torch.float32, device=optimizer.device).clone()
    vertex_coord[..., 0] /= float(inference.image_width)
    vertex_coord[..., 1] /= float(inference.image_height)
    return vertex_coord, vertex_depth


def _register_pose(
    optimizer,
    camera_view_mat: torch.Tensor,
    camera_viewproj_mat: torch.Tensor,
    vertex_coord: torch.Tensor,
    vertex_depth: torch.Tensor,
    v_canonical: torch.Tensor,
    j_canonical: torch.Tensor,
) -> torch.Tensor:
    x = torch.zeros((optimizer.dim_x,), dtype=torch.float32, device=optimizer.device)
    for _ in range(int(optimizer.config.pose_registration_iterations)):
        flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = optimizer.solver_ops.expression_jacobian(
            x,
            v_canonical,
            j_canonical,
            optimizer.expr_dirs,
            optimizer.pose_dirs,
            optimizer.lbs_weights,
        )
        residual, jacobian = optimizer.solver_ops.assemble_x_system(
            camera_view_mat,
            camera_viewproj_mat,
            flame_v,
            flame_j,
            vertex_coord,
            vertex_depth,
            None,
            None,
            x,
            flame_v_jacobian,
            flame_j_jacobian,
            alignment_weight=1.0,
            rel_depth_weight=0.0,
            exp_reg_weight=0.0,
            pose_reg_weight=0.0,
            vertex_indices=optimizer.key_vertex_ids,
        )
        x[optimizer._opt_pose_dims] = x[optimizer._opt_pose_dims] + optimizer.solver_ops.solve_normal_equation(
            optimizer.solver_context,
            jacobian[:, optimizer._opt_pose_dims],
            residual,
            optimizer.config.pose_damping,
        )
    return x


def _register_candidate(
    optimizer,
    camera_view_mat: torch.Tensor,
    camera_viewproj_mat: torch.Tensor,
    vertex_coord: torch.Tensor,
    vertex_depth: torch.Tensor,
    init_x: torch.Tensor,
    num_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.zeros((300,), dtype=torch.float32, device=optimizer.device)
    x = init_x.reshape(optimizer.dim_x).clone()

    for _ in range(int(num_iters)):
        v_canonical, j_canonical = optimizer._calc_canonical_flame(identity)

        flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = optimizer.solver_ops.expression_jacobian(
            x,
            v_canonical,
            j_canonical,
            optimizer.expr_dirs,
            optimizer.pose_dirs,
            optimizer.lbs_weights,
        )
        residual, jacobian = optimizer.solver_ops.assemble_x_system(
            camera_view_mat,
            camera_viewproj_mat,
            flame_v,
            flame_j,
            vertex_coord,
            vertex_depth,
            None,
            None,
            x,
            flame_v_jacobian,
            flame_j_jacobian,
            optimizer.config.correspondence_weight,
            0.0,
            optimizer.config.expression_regularization,
            optimizer.config.pose_regularization,
            optimizer.key_vertex_ids,
        )
        x = x + optimizer.solver_ops.solve_normal_equation(
            optimizer.solver_context,
            jacobian,
            residual,
            optimizer.config.damping,
        )

        flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = optimizer.solver_ops.identity_jacobian(
            x,
            v_canonical,
            j_canonical,
            optimizer.joints_dirs,
            optimizer.shape_dirs,
            optimizer.expr_dirs,
            optimizer.pose_dirs,
            optimizer.lbs_weights,
        )
        residual, jacobian = optimizer.solver_ops.assemble_identity_system(
            camera_view_mat,
            camera_viewproj_mat,
            flame_v,
            flame_j,
            vertex_coord,
            vertex_depth,
            None,
            None,
            identity,
            flame_v_jacobian,
            flame_j_jacobian,
            optimizer.config.correspondence_weight,
            0.0,
            optimizer.config.identity_regularization,
            optimizer.key_vertex_ids,
        )
        identity = identity + optimizer.solver_ops.solve_normal_equation(
            optimizer.solver_context,
            jacobian,
            residual,
            optimizer.config.damping,
        )

    return identity, x


def _evaluate_candidate_energy(
    optimizer,
    camera_view_mat: torch.Tensor,
    camera_viewproj_mat: torch.Tensor,
    vertex_coord: torch.Tensor,
    vertex_depth: torch.Tensor,
    identity: torch.Tensor,
    x: torch.Tensor,
) -> float:
    v_canonical, j_canonical = optimizer._calc_canonical_flame(identity)
    flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = optimizer.solver_ops.expression_jacobian(
        x,
        v_canonical,
        j_canonical,
        optimizer.expr_dirs,
        optimizer.pose_dirs,
        optimizer.lbs_weights,
    )
    residual, _ = optimizer.solver_ops.assemble_x_system(
        camera_view_mat,
        camera_viewproj_mat,
        flame_v,
        flame_j,
        vertex_coord,
        vertex_depth,
        None,
        None,
        x,
        flame_v_jacobian,
        flame_j_jacobian,
        optimizer.config.correspondence_weight,
        0.0,
        optimizer.config.expression_regularization,
        optimizer.config.pose_regularization,
        optimizer.key_vertex_ids,
    )

    num_views = int(camera_view_mat.shape[0])
    num_items = int(optimizer.key_vertex_ids.numel()) if optimizer.key_vertex_ids.numel() > 0 else int(vertex_coord.shape[-2])
    obs_dim = 2
    num_data_rows = num_views * num_items * obs_dim

    if residual.ndim == 1:
        data_residual = residual[:num_data_rows].reshape(-1, obs_dim)
        return float(torch.square(data_residual).sum().item())
    if residual.ndim == 2:
        data_residual = residual[:, :num_data_rows].reshape(residual.shape[0], -1, obs_dim)
        return float(torch.square(data_residual).sum().item())
    raise ValueError("x residual must be [R] or [T,R].")


def search_online_camera_fov_y(optimizer, inference, base_camera: Camera) -> float:
    vertex_coord, vertex_depth = _normalize_single_frame_observations(
        optimizer,
        inference,
    )
    pose_identity = torch.zeros((300,), dtype=torch.float32, device=optimizer.device)
    pose_v_canonical, pose_j_canonical = optimizer._calc_canonical_flame(pose_identity)

    def evaluate_fov(fov_y_degrees: float) -> float:
        camera = _build_candidate_camera(base_camera, fov_y_degrees)
        camera_view_mat, camera_viewproj_mat = _single_camera_matrices(optimizer, camera)
        pose_x = _register_pose(
            optimizer,
            camera_view_mat,
            camera_viewproj_mat,
            vertex_coord,
            vertex_depth,
            pose_v_canonical,
            pose_j_canonical,
        )
        identity, x = _register_candidate(
            optimizer,
            camera_view_mat,
            camera_viewproj_mat,
            vertex_coord,
            vertex_depth,
            pose_x,
            num_iters=int(optimizer.config.fov_registration_iterations),
        )
        return _evaluate_candidate_energy(
            optimizer,
            camera_view_mat,
            camera_viewproj_mat,
            vertex_coord,
            vertex_depth,
            identity,
            x,
        )

    l = float(optimizer.config.fov_y_min)
    r = float(optimizer.config.fov_y_max)
    phi = 0.5 * (math.sqrt(5.0) - 1.0)

    x1 = r - phi * (r - l)
    x2 = l + phi * (r - l)
    e1 = evaluate_fov(x1)
    e2 = evaluate_fov(x2)

    for _ in range(int(optimizer.config.fov_search_iterations)):
        if r - l < float(optimizer.config.fov_search_epsilon):
            break
        if e1 < e2:
            r = x2
            x2 = x1
            e2 = e1
            x1 = r - phi * (r - l)
            e1 = evaluate_fov(x1)
        else:
            l = x1
            x1 = x2
            e1 = e2
            x2 = l + phi * (r - l)
            e2 = evaluate_fov(x2)

    return 0.5 * (l + r)

