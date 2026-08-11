from __future__ import annotations

import math
import pickle

import numpy as np
import torch

from camera import Camera
from flame_solver import FlameSolverOps, SolverContext
from common.config import OnlineGNOptimizerConfig, FLAMEConfig
from common.types import InferenceOutput, OnlineInferenceOutput
from common.utils import load_expression_blendshapes
from .gn_flame_optimizer import EYE_REGION_VERTICES
from .online_fov_search import search_online_camera_fov_y


def rotvec_to_rotmat_numpy(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32)
    if rotvec.ndim == 1:
        rotvec = rotvec.reshape(1, 3)
        squeeze = True
    elif rotvec.ndim == 2 and rotvec.shape[1] == 3:
        squeeze = False
    else:
        raise ValueError(f"rotvec_to_rotmat_numpy expects shape [3] or [N, 3], got {rotvec.shape}.")

    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = np.zeros_like(rotvec)
    valid = angle[:, 0] > 1e-8
    axis[valid] = rotvec[valid] / angle[valid]

    kx = axis[:, 0]
    ky = axis[:, 1]
    kz = axis[:, 2]
    zeros = np.zeros_like(kx)
    K = np.stack(
        [
            np.stack([zeros, -kz, ky], axis=1),
            np.stack([kz, zeros, -kx], axis=1),
            np.stack([-ky, kx, zeros], axis=1),
        ],
        axis=1,
    )

    eye = np.broadcast_to(np.eye(3, dtype=np.float32), (rotvec.shape[0], 3, 3)).copy()
    sin_theta = np.sin(angle).astype(np.float32)[:, None]
    cos_theta = np.cos(angle).astype(np.float32)[:, None]
    outer = axis[:, :, None] * axis[:, None, :]
    rot = cos_theta * eye + (1.0 - cos_theta) * outer + sin_theta * K
    rot[~valid] = eye[~valid]
    return rot[0] if squeeze else rot


class KeyframeBuffer:
    def __init__(self, capacity: int, dim_x: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.valid_count = 0
        self.vertex_coord = torch.empty((self.capacity, 5023, 2), dtype=torch.float32, device=device)
        self.vertex_depth = torch.empty((self.capacity, 5023, 1), dtype=torch.float32, device=device)
        self.vertex_coord_log_var = torch.empty((self.capacity, 5023, 1), dtype=torch.float32, device=device)
        self.x = torch.empty((self.capacity, int(dim_x)), dtype=torch.float32, device=device)
        self.head_rot = np.empty((self.capacity, 3, 3), dtype=np.float32)

    def reset(self) -> None:
        self.valid_count = 0

    def append(
        self,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        x: torch.Tensor,
        head_rot: np.ndarray,
    ) -> int:
        if self.valid_count >= self.capacity:
            raise ValueError("KeyframeBuffer is full.")
        slot = self.valid_count
        self.vertex_coord[slot].copy_(vertex_coord)
        self.vertex_depth[slot].copy_(vertex_depth)
        self.vertex_coord_log_var[slot].copy_(vertex_coord_log_var)
        self.x[slot].copy_(x)
        self.head_rot[slot] = head_rot
        self.valid_count += 1
        return slot

    def replace(
        self,
        slot: int,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        x: torch.Tensor,
        head_rot: np.ndarray,
    ) -> int:
        if slot < 0 or slot >= self.valid_count:
            raise ValueError("Invalid keyframe slot.")
        self.vertex_coord[slot].copy_(vertex_coord)
        self.vertex_depth[slot].copy_(vertex_depth)
        self.vertex_coord_log_var[slot].copy_(vertex_coord_log_var)
        self.x[slot].copy_(x)
        self.head_rot[slot] = head_rot
        return slot

    def get_active_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.valid_count == 0:
            raise ValueError("KeyframeBuffer is empty.")
        return (
            self.vertex_coord[: self.valid_count],
            self.vertex_depth[: self.valid_count],
            self.vertex_coord_log_var[: self.valid_count],
            self.x[: self.valid_count],
        )

    def get_active_head_rot(self) -> np.ndarray:
        if self.valid_count == 0:
            raise ValueError("KeyframeBuffer is empty.")
        return self.head_rot[: self.valid_count]

    def update_x(self, xs_active: torch.Tensor, head_rot_active: np.ndarray) -> None:
        self.x[: self.valid_count].copy_(xs_active)
        self.head_rot[: self.valid_count] = head_rot_active


class OnlineGNOptimizer:
    def __init__(self, config: OnlineGNOptimizerConfig):
        self.config = config
        self.device = torch.device(config.device)
        flame_config = config.flame
        self.num_expressions = flame_config.resolved_num_expressions()
        self.dim_x = self.num_expressions + 18
        self.pose_start = self.num_expressions
        self.trans_start = self.num_expressions + 15
        self.solver_ops = FlameSolverOps(num_expressions=self.num_expressions)
        self.solver_context = SolverContext.create(self.solver_ops.extension)

        self._load_flame_assets(flame_config)
        self._eye_region_ids = torch.tensor(EYE_REGION_VERTICES, dtype=torch.long, device=self.device)
        self._eye_boost_log_var_delta = -2.0 * math.log(max(float(config.eye_region_boost_factor), 1e-8))
        self._opt_pose_dims = torch.tensor(
            [
                self.pose_start,
                self.pose_start + 1,
                self.pose_start + 2,
                self.trans_start,
                self.trans_start + 1,
                self.trans_start + 2,
            ],
            dtype=torch.int64,
            device=self.device,
        )

        self._camera_view_mat: torch.Tensor | None = None
        self._camera_viewproj_mat: torch.Tensor | None = None

        self._keyframe_buffer = KeyframeBuffer(
            capacity=int(config.max_keyframes),
            dim_x=self.dim_x,
            device=self.device,
        )

        self._frame_id = 0
        self._last_x: torch.Tensor | None = None
        self._pending_refine_budget = 0
        self._identity = torch.zeros((300,), dtype=torch.float32, device=self.device)
        self._v_canonical, self._j_canonical = self._calc_canonical_flame(self._identity)
        self._update_identity_callback = None

    def set_identity_callback(self, func):
        self._update_identity_callback = func

    def refresh_camera_matrices(self, camera: Camera) -> None:
        self._camera_view_mat = camera.get_w2v.to(self.device, torch.float32).unsqueeze(0)
        self._camera_viewproj_mat = camera.get_full_proj.to(self.device, torch.float32).unsqueeze(0)

    @property
    def identity(self) -> torch.Tensor:
        return self._identity.detach().clone()

    @property
    def canonical(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._v_canonical.detach(), self._j_canonical.detach()

    def reset(self):
        self._frame_id = 0
        self._last_x = None
        self._pending_refine_budget = 0
        self._keyframe_buffer.reset()
        self._identity.zero_()
        self._v_canonical, self._j_canonical = self._calc_canonical_flame(self._identity)
        if self._update_identity_callback is not None:
            self._update_identity_callback(self._v_canonical, self._j_canonical, self._identity)

    def register_camera_fov(self, inference: InferenceOutput | OnlineInferenceOutput, camera: Camera) -> float:
        return search_online_camera_fov_y(self, inference, camera)

    def _load_flame_assets(self, flame_config: FLAMEConfig) -> None:
        with open(flame_config.flame_model_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self.v_template = torch.from_numpy(np.asarray(data["v_template"])).to(dtype=torch.float32, device=self.device)
        self.faces = torch.from_numpy(np.asarray(data["f"]).astype(np.int64)).to(dtype=torch.int64, device=self.device)
        self.pose_dirs = torch.from_numpy(np.asarray(data["posedirs"])).to(dtype=torch.float32, device=self.device)
        self.lbs_weights = torch.from_numpy(np.asarray(data["weights"])).to(dtype=torch.float32, device=self.device)

        j_regressor_coo = data["J_regressor"].tocoo()
        self.J_regressor_row = torch.from_numpy(np.asarray(j_regressor_coo.row)).to(dtype=torch.int32, device=self.device)
        self.J_regressor_col = torch.from_numpy(np.asarray(j_regressor_coo.col)).to(dtype=torch.int32, device=self.device)
        self.J_regressor_values = torch.from_numpy(np.asarray(j_regressor_coo.data)).to(dtype=torch.float32, device=self.device)

        num_pose_basis = self.pose_dirs.shape[-1]
        shapedirs_np = np.asarray(data["shapedirs"]).astype(np.float32, copy=False)
        self.shape_dirs = torch.from_numpy(np.ascontiguousarray(shapedirs_np[..., :300])).to(
            dtype=torch.float32, device=self.device
        ).view(-1, 300).T.contiguous()
        expr_dirs_np = load_expression_blendshapes(data, flame_config)
        self.expr_dirs = torch.from_numpy(expr_dirs_np).to(dtype=torch.float32, device=self.device) \
            .view(-1, self.num_expressions).T.contiguous()
        self.pose_dirs = self.pose_dirs.reshape([-1, num_pose_basis]).T.contiguous()

        j_regressor_dense = torch.from_numpy(np.asarray(data["J_regressor"].todense())).to(dtype=torch.float32, device=self.device)
        self.joints_dirs = (
            j_regressor_dense @ self.shape_dirs.view(300, 5023, 3).permute(1, 2, 0).reshape(5023, 300 * 3)
        ).view(5, 3, 300).permute(2, 0, 1).contiguous()

        # enable key vertices
        # self.key_vertex_ids = np.unique(np.load(flame_config.flame_assets_path)['key_region_faces'])
        # self.key_vertex_ids = torch.from_numpy(self.key_vertex_ids).to(dtype=torch.int64, device=self.device)
        self.key_vertex_ids = torch.empty((0,), dtype=torch.int64, device=self.device)

    def _prepare_observation(
        self,
        inference: InferenceOutput | OnlineInferenceOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vertex_coord = torch.as_tensor(inference.vertex_coord, dtype=torch.float32, device=self.device).clone()
        vertex_depth = torch.as_tensor(inference.vertex_depth, dtype=torch.float32, device=self.device).clone()
        vertex_coord_log_var = torch.as_tensor(inference.vertex_coord_log_var, dtype=torch.float32, device=self.device).clone()
        vertex_coord[..., 0] /= float(inference.image_width)
        vertex_coord[..., 1] /= float(inference.image_height)
        if self.config.boost_eye_region:
            vertex_coord_log_var[..., self._eye_region_ids, :] += self._eye_boost_log_var_delta
        return vertex_coord, vertex_depth, vertex_coord_log_var

    def _calc_canonical_flame(self, identity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.solver_ops.calc_canonical(
            identity,
            self.v_template,
            self.shape_dirs,
            self.J_regressor_row,
            self.J_regressor_col,
            self.J_regressor_values,
        )

    def _calc_posed_flame(
        self,
        v_canonical: torch.Tensor,
        j_canonical: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.solver_ops.forward_vertices(
            x,
            v_canonical,
            j_canonical,
            self.expr_dirs,
            self.pose_dirs,
            self.lbs_weights,
        )

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(self.dim_x).to(device=self.device, dtype=torch.float32)
        return self._calc_posed_flame(self._v_canonical, self._j_canonical, x)

    @torch.no_grad()
    def decode_with_identity(self, identity: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        identity = identity.reshape(300).to(device=self.device, dtype=torch.float32)
        x = x.reshape(self.dim_x).to(device=self.device, dtype=torch.float32)
        v_canonical, j_canonical = self._calc_canonical_flame(identity)
        return self._calc_posed_flame(v_canonical, j_canonical, x)

    def _extract_head_rot(self, x: torch.Tensor) -> np.ndarray:
        if x.ndim == 1:
            pose_rotvec = x[self.pose_start:self.pose_start + 6].detach().cpu().numpy().astype(np.float32, copy=False).reshape(2, 3)
            pose_rot = rotvec_to_rotmat_numpy(pose_rotvec)
            return np.matmul(pose_rot[0], pose_rot[1]).astype(np.float32, copy=False)
        if x.ndim == 2:
            pose_rotvec = x[:, self.pose_start:self.pose_start + 6].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 2, 3)
            pose_rot = rotvec_to_rotmat_numpy(pose_rotvec.reshape(-1, 3)).reshape(-1, 2, 3, 3)
            return np.matmul(pose_rot[:, 0], pose_rot[:, 1]).astype(np.float32, copy=False)
        raise ValueError("Unsupported x shape for head rotation extraction.")

    @staticmethod
    def _rotation_angle_to_active(active_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
        rel_rot = np.matmul(active_rot, current_rot.T[None, :, :])
        trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return np.arccos(cos_theta).astype(np.float32, copy=False)

    @staticmethod
    def _rotation_angle_matrix(active_rot: np.ndarray) -> np.ndarray:
        rel_rot = np.matmul(active_rot[:, None, :, :], np.swapaxes(active_rot[None, :, :, :], -1, -2))
        trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        return np.arccos(cos_theta).astype(np.float32, copy=False)

    def _compute_coverage_objective(self, head_rot: np.ndarray) -> tuple[float, int]:
        if head_rot.shape[0] <= 1:
            return 0.0, 0
        dist = self._rotation_angle_matrix(head_rot)
        np.fill_diagonal(dist, np.inf)
        nn_dist = dist.min(axis=1)
        return float(nn_dist.mean()), int(nn_dist.argmin())

    def _try_update_keyframe_buffer(
        self,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        x: torch.Tensor,
    ) -> bool:
        head_rot = self._extract_head_rot(x)
        if self._keyframe_buffer.valid_count == 0:
            self._keyframe_buffer.append(vertex_coord, vertex_depth, vertex_coord_log_var, x, head_rot)
            return True

        active_head_rot = self._keyframe_buffer.get_active_head_rot()
        head_distance = float(self._rotation_angle_to_active(active_head_rot, head_rot).min())

        if self._keyframe_buffer.valid_count < self._keyframe_buffer.capacity:
            if head_distance < float(self.config.keyframe_head_threshold):
                return False
            self._keyframe_buffer.append(vertex_coord, vertex_depth, vertex_coord_log_var, x, head_rot)
            return True

        before_objective, redundant_idx = self._compute_coverage_objective(active_head_rot.copy())
        candidate_head_rot = active_head_rot.copy()
        candidate_head_rot[redundant_idx] = head_rot
        after_objective, _ = self._compute_coverage_objective(candidate_head_rot)
        if after_objective <= before_objective:
            return False
        self._keyframe_buffer.replace(redundant_idx, vertex_coord, vertex_depth, vertex_coord_log_var, x, head_rot)
        return True

    def _register_pose(
        self,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        num_iters: int | None = None,
    ) -> torch.Tensor:
        if self._camera_view_mat is None or self._camera_viewproj_mat is None:
            raise RuntimeError("Camera matrices are not initialized. Call refresh_camera_matrices(camera) first.")
        camera_view_mat = self._camera_view_mat
        camera_viewproj_mat = self._camera_viewproj_mat
        x = torch.zeros((self.dim_x,), dtype=torch.float32, device=self.device)
        effective_num_iters = self.config.pose_registration_iterations if num_iters is None else int(num_iters)
        for _ in range(effective_num_iters):
            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x,
                self._v_canonical,
                self._j_canonical,
                self.expr_dirs,
                self.pose_dirs,
                self.lbs_weights,
            )
            residual, jacobian = self.solver_ops.assemble_x_system(
                camera_view_mat,
                camera_viewproj_mat,
                flame_v,
                flame_j,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_coord_log_var,
                x,
                flame_v_jacobian,
                flame_j_jacobian,
                alignment_weight=1.0,
                rel_depth_weight=0.0,
                exp_reg_weight=0.0,
                pose_reg_weight=0.0,
                vertex_indices=self.key_vertex_ids,
            )
            x[self._opt_pose_dims] = x[self._opt_pose_dims] + self.solver_ops.solve_normal_equation(
                self.solver_context,
                jacobian[:, self._opt_pose_dims],
                residual,
                self.config.pose_damping,
            )
        return x

    def _register(
        self,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        init_x: torch.Tensor,
        num_iters: int,
    ) -> torch.Tensor:
        if self._camera_view_mat is None or self._camera_viewproj_mat is None:
            raise RuntimeError("Camera matrices are not initialized. Call refresh_camera_matrices(camera) first.")
        camera_view_mat = self._camera_view_mat
        camera_viewproj_mat = self._camera_viewproj_mat
        if vertex_coord.ndim == 2 and vertex_depth.ndim == 2:
            x = init_x.reshape(self.dim_x).clone()
        elif vertex_coord.ndim == 3 and vertex_depth.ndim == 3:
            x = init_x.reshape(vertex_coord.shape[0], self.dim_x).clone()
        else:
            raise ValueError("Invalid shape of vertex_coord or vertex_depth.")

        for _ in range(int(num_iters)):
            v_canonical, j_canonical = self._calc_canonical_flame(self._identity)

            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x,
                v_canonical,
                j_canonical,
                self.expr_dirs,
                self.pose_dirs,
                self.lbs_weights,
            )
            residual, jacobian = self.solver_ops.assemble_x_system(
                camera_view_mat,
                camera_viewproj_mat,
                flame_v,
                flame_j,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_coord_log_var,
                x,
                flame_v_jacobian,
                flame_j_jacobian,
                self.config.correspondence_weight,
                0.0,
                self.config.expression_regularization,
                self.config.pose_regularization,
                self.key_vertex_ids,
            )
            x = x + self.solver_ops.solve_normal_equation(
                self.solver_context,
                jacobian,
                residual,
                self.config.damping,
            )

            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.identity_jacobian(
                x,
                v_canonical,
                j_canonical,
                self.joints_dirs,
                self.shape_dirs,
                self.expr_dirs,
                self.pose_dirs,
                self.lbs_weights,
            )
            residual, jacobian = self.solver_ops.assemble_identity_system(
                camera_view_mat,
                camera_viewproj_mat,
                flame_v,
                flame_j,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_coord_log_var,
                self._identity,
                flame_v_jacobian,
                flame_j_jacobian,
                self.config.correspondence_weight,
                0.0,
                self.config.identity_regularization,
                self.key_vertex_ids,
            )
            self._identity = self._identity + self.solver_ops.solve_normal_equation(
                self.solver_context,
                jacobian,
                residual,
                self.config.damping,
            )

        self._v_canonical, self._j_canonical = self._calc_canonical_flame(self._identity)
        if self._update_identity_callback is not None:
            self._update_identity_callback(self._v_canonical, self._j_canonical, self._identity)
        return x

    def _track(
        self,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor,
        init_x: torch.Tensor,
        num_iters: int | None = None,
    ) -> torch.Tensor:
        if self._camera_view_mat is None or self._camera_viewproj_mat is None:
            raise RuntimeError("Camera matrices are not initialized. Call refresh_camera_matrices(camera) first.")
        camera_view_mat = self._camera_view_mat
        camera_viewproj_mat = self._camera_viewproj_mat
        x = init_x.reshape(self.dim_x).clone()
        effective_num_iters = self.config.tracking_iterations if num_iters is None else int(num_iters)
        prev_energy = float("inf")
        for it in range(effective_num_iters):
            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x,
                self._v_canonical,
                self._j_canonical,
                self.expr_dirs,
                self.pose_dirs,
                self.lbs_weights,
            )
            residual, jacobian = self.solver_ops.assemble_x_system(
                camera_view_mat,
                camera_viewproj_mat,
                flame_v,
                flame_j,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_coord_log_var,
                x,
                flame_v_jacobian,
                flame_j_jacobian,
                self.config.correspondence_weight,
                0.0,
                self.config.expression_regularization,
                self.config.pose_regularization,
                self.key_vertex_ids,
            )

            energy = float(torch.sum(residual ** 2).item())
            if prev_energy - energy < self.config.tracking_early_stop_ratio * prev_energy:
                break
            prev_energy = energy

            x = x + self.solver_ops.solve_normal_equation(
                self.solver_context,
                jacobian,
                residual,
                self.config.damping,
            )
        # print(f"Track iters: {it + 1}, final energy: {prev_energy:.4f}")
        return x

    def track(self, inference: InferenceOutput | OnlineInferenceOutput):
        vertex_coord, vertex_depth, vertex_coord_log_var = self._prepare_observation(inference)

        buffer_updated = False
        if self._frame_id == 0:
            x = self._register_pose(vertex_coord, vertex_depth, vertex_coord_log_var)
            x = self._register(
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                init_x=x,
                num_iters=int(self.config.registration_iterations),
            )
            head_rot = self._extract_head_rot(x)
            self._keyframe_buffer.append(vertex_coord, vertex_depth, vertex_coord_log_var, x, head_rot)
            self._pending_refine_budget += int(self.config.identity_refinement_steps)
        else:
            if self._last_x is None:
                raise ValueError("last_x is missing for non-first frame.")
            x = self._track(vertex_coord, vertex_depth, vertex_coord_log_var, init_x=self._last_x)
            interval = int(self.config.keyframe_interval)
            if interval > 0 and self._frame_id % interval == 0:
                buffer_updated = self._try_update_keyframe_buffer(
                    vertex_coord,
                    vertex_depth,
                    vertex_coord_log_var,
                    x,
                )
                if buffer_updated:
                    self._pending_refine_budget += int(self.config.identity_refinement_steps)

            if not buffer_updated and self._pending_refine_budget > 0 and self._keyframe_buffer.valid_count > 0:
                (
                    keyframe_vertex_coord,
                    keyframe_vertex_depth,
                    keyframe_vertex_coord_log_var,
                    xs,
                ) = self._keyframe_buffer.get_active_tensors()
                xs = self._register(
                    keyframe_vertex_coord,
                    keyframe_vertex_depth,
                    keyframe_vertex_coord_log_var,
                    init_x=xs,
                    num_iters=1,
                )
                active_head_rot = self._extract_head_rot(xs)
                self._keyframe_buffer.update_x(xs, active_head_rot)
                self._pending_refine_budget -= 1

        self._frame_id += 1
        self._last_x = x.clone()
        return x

