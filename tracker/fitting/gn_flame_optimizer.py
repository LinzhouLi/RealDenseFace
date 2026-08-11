from __future__ import annotations

import math
import pickle
from typing import Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from camera import Camera, PerspectiveCamera
from camera.camera import focal2fov, fov2focal
from flame_solver import FlameSolverOps, SolverContext
from common.config import INVISIBLE_LOG_VAR, GNFlameOptimizerConfig
from common.utils import load_expression_blendshapes, load_mesh_from_obj
from common.types import FittingOutput, InferenceOutput


EYE_REGION_VERTICES = [
    # eyelid
    807, 814, 822, 824, 827, 991, 994, 995, 1093, 1096, 1113, 1170, 1175, 1195, 1200, 1201, 1202, 1216, 1218, 1329, 1331, 1336, 1340, 1343, 1344, 1354, 1355, 1357, 3833, 3855,
    2264, 2267, 2271, 2273, 2276, 2355, 2358, 2359, 2402, 2403, 2406, 2436, 2437, 2447, 2448, 2449, 2450, 2451, 2453, 2485, 2486, 2487, 2491, 2494, 2495, 2505, 2506, 2508, 3632, 3689,
    # iris
    4478, 4482, 4486, 4490, 4494, 4498, 4502, 4506, 4510, 4514, 4518, 4522, 4526, 4530, 4534, 4538, 4542, 4546, 4550, 4554, 4558, 4562, 4566, 4570, 4574, 4578, 4582, 4586, 4590, 4594, 4599, 4603,
    3932, 3936, 3940, 3944, 3948, 3952, 3956, 3960, 3964, 3968, 3972, 3976, 3980, 3984, 3988, 3992, 3996, 4000, 4004, 4008, 4012, 4016, 4020, 4024, 4028, 4032, 4036, 4040, 4044, 4048, 4053, 4057
]


class GNFlameOptimizer:
    HISTORY_COMPONENTS = ("align_loss", "depth_loss", "reg_expr", "reg_pose", "reg_ident", "total_loss")

    def __init__(self, config: GNFlameOptimizerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_flame_assets(config.flame)
        self.solver_ops = FlameSolverOps(num_expressions=self.num_expressions)
        self.solver_context = SolverContext.create(self.solver_ops.extension)
        self.key_vertex_ids = torch.tensor([], dtype=torch.long, device=self.device)
        # key_vertex_ids = np.unique(load_mesh_from_obj("weights/flame/flame_key_region.obj")[1].flatten())
        # self.key_vertex_ids = torch.from_numpy(key_vertex_ids).to(dtype=torch.int64, device=self.device)
        self._empty_uncertainty = torch.empty((0,), dtype=torch.float32, device=self.device)
        self._eye_region_ids = torch.tensor(EYE_REGION_VERTICES, dtype=torch.long, device=self.device)
        self._eye_boost_log_var_delta = -2.0 * math.log(max(float(config.eye_region_boost_factor), 1e-8))

    def _load_flame_assets(self, flame_config) -> None:
        self.num_expressions = flame_config.resolved_num_expressions()
        self.dim_x = self.num_expressions + 18
        self.pose_start = self.num_expressions
        self.trans_start = self.num_expressions + 15

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

        expr_dirs_np = load_expression_blendshapes(data, flame_config)  # (V, 3, num_expressions)
        self.expr_dirs = torch.from_numpy(expr_dirs_np).to(dtype=torch.float32, device=self.device) \
            .view(-1, self.num_expressions).T.contiguous()
        self.pose_dirs = self.pose_dirs.reshape([-1, num_pose_basis]).T.contiguous()

        j_regressor_dense = torch.from_numpy(np.asarray(data["J_regressor"].todense())).to(dtype=torch.float32, device=self.device)
        self.joints_dirs = (
            j_regressor_dense @ self.shape_dirs.view(300, 5023, 3).permute(1, 2, 0).reshape(5023, 300 * 3)
        ).view(5, 3, 300).permute(2, 0, 1).contiguous()

    def _apply_eye_region_boost(self, vertex_coord_log_var: torch.Tensor | None) -> torch.Tensor | None:
        if vertex_coord_log_var is None or not self.config.boost_eye_region:
            return vertex_coord_log_var
        vertex_coord_log_var = vertex_coord_log_var.clone()
        vertex_coord_log_var[..., self._eye_region_ids, :] += self._eye_boost_log_var_delta
        return vertex_coord_log_var

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

    @staticmethod
    def _check_finite(tensor: torch.Tensor | None, label: str) -> bool:
        """Return True if *tensor* is fully finite. Otherwise print a NaN/inf diagnostic
        line including stats over the finite slice, and return False. ``None`` is
        treated as finite (nothing to check)."""
        if tensor is None or not torch.is_tensor(tensor):
            return True
        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all().item()):
            return True
        bad = (~finite_mask)
        n_bad = int(bad.sum().item())
        n_total = int(tensor.numel())
        n_nan = int(torch.isnan(tensor).sum().item())
        n_pinf = int(torch.isposinf(tensor).sum().item())
        n_ninf = int(torch.isneginf(tensor).sum().item())
        finite_vals = tensor[finite_mask].detach().to(torch.float32)
        if finite_vals.numel() > 0:
            amin = float(finite_vals.min().item())
            amax = float(finite_vals.max().item())
            amean = float(finite_vals.mean().item())
            stats = f"finite[min={amin:.4g} max={amax:.4g} mean={amean:.4g}]"
        else:
            stats = "no finite values"
        print(
            f"[GN][NaN!!] {label}: bad={n_bad}/{n_total} "
            f"(nan={n_nan}, +inf={n_pinf}, -inf={n_ninf}) "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype}; {stats}",
            flush=True,
        )
        return False

    def _stack_camera_matrices(self, cameras: Sequence[Camera]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(cameras) == 0:
            raise ValueError("At least one camera is required.")
        view_mats = [camera.get_w2v for camera in cameras]
        viewproj_mats = [camera.get_full_proj for camera in cameras]
        return (
            torch.stack(view_mats, dim=0).to(self.device, torch.float32), 
            torch.stack(viewproj_mats, dim=0).to(self.device, torch.float32)
        )

    def _single_camera_matrices(self, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        view_mat = camera.get_w2v.to(self.device, torch.float32).unsqueeze(0)
        viewproj_mat = camera.get_full_proj.to(self.device, torch.float32).unsqueeze(0)
        return view_mat, viewproj_mat

    def _build_config_camera(self, image_width: int, image_height: int, fov_y: float) -> PerspectiveCamera:
        rot = self.config.camera_rotation.reshape(3, 3)
        pos = self.config.camera_position.reshape(3)
        return PerspectiveCamera(
            fov_y=math.radians(float(fov_y)),
            rot=rot,
            pos=pos,
            width=int(image_width),
            height=int(image_height),
        )

    def _fov_y_to_focal(self, image_height: int, fov_y: float) -> float:
        return fov2focal(math.radians(float(fov_y)), float(image_height))

    def _focal_to_fov_y(self, image_height: int, focal_y: float) -> float:
        return math.degrees(focal2fov(float(focal_y), float(image_height)))

    def _evaluate_energy(
        self,
        camera_view_mat: torch.Tensor,
        camera_viewproj_mat: torch.Tensor,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor | None,
        vertex_depth_log_var: torch.Tensor | None,
        identity: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        num_views = int(camera_view_mat.shape[0])

        identity = identity.to(device=self.device, dtype=torch.float32).reshape(300)
        x = x.to(device=self.device, dtype=torch.float32)
        vertex_indices = self.key_vertex_ids
        num_items = int(vertex_indices.numel()) if vertex_indices.numel() > 0 else int(vertex_coord.shape[-2])
        # CUDA assemble_jacobian writes obs_dim columns per item: 3 (uv + depth)
        # when depth_weight > 0, else 2 (uv only). See assemble_jacobian.cu:39, 205.
        obs_dim = 3 if float(self.config.relative_depth_weight) > 0.0 else 2
        num_data_rows = num_views * num_items * obs_dim
        chunk_size = int(self.config.energy_chunk_size)

        def energy_from_residual(residual: torch.Tensor) -> torch.Tensor:
            if residual.ndim == 1:
                data_residual = residual[:num_data_rows].reshape(-1, obs_dim)
                return torch.square(data_residual).sum()
            if residual.ndim == 2:
                data_residual = residual[:, :num_data_rows].reshape(residual.shape[0], -1, obs_dim)
                return torch.square(data_residual).sum(dim=(1, 2))
            raise ValueError("x residual must be [R] or [T,R].")

        v_canonical, j_canonical = self._calc_canonical_flame(identity)
        if x.ndim == 1:
            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x,
                v_canonical,
                j_canonical,
                self.expr_dirs,
                self.pose_dirs,
                self.lbs_weights,
            )
            residual, _ = self.solver_ops.assemble_x_system(
                camera_view_mat,
                camera_viewproj_mat,
                flame_v,
                flame_j,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_depth_log_var,
                x,
                flame_v_jacobian,
                flame_j_jacobian,
                self.config.correspondence_weight,
                self.config.relative_depth_weight,
                self.config.expression_regularization,
                self.config.pose_regularization,
                vertex_indices,
            )
            return energy_from_residual(residual)

        if chunk_size <= 0:
            raise ValueError("sequence_energy_chunk_size must be positive.")

        batch_size = int(x.shape[0])
        chunk_energies: list[torch.Tensor] = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            x_chunk = x[start:end]
            vertex_coord_chunk = vertex_coord[start:end]
            vertex_depth_chunk = vertex_depth[start:end]
            if vertex_coord_log_var is None or vertex_depth_log_var is None:
                vertex_coord_log_var_chunk = vertex_coord_log_var
                vertex_depth_log_var_chunk = vertex_depth_log_var
            else:
                vertex_coord_log_var_chunk = vertex_coord_log_var[start:end]
                vertex_depth_log_var_chunk = vertex_depth_log_var[start:end]

            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x_chunk,
                v_canonical,
                j_canonical,
                self.expr_dirs,
                self.pose_dirs,
                self.lbs_weights,
            )
            residual, _ = self.solver_ops.assemble_x_system(
                camera_view_mat,
                camera_viewproj_mat,
                flame_v,
                flame_j,
                vertex_coord_chunk,
                vertex_depth_chunk,
                vertex_coord_log_var_chunk,
                vertex_depth_log_var_chunk,
                x_chunk,
                flame_v_jacobian,
                flame_j_jacobian,
                self.config.correspondence_weight,
                self.config.relative_depth_weight,
                self.config.expression_regularization,
                self.config.pose_regularization,
                vertex_indices,
            )
            chunk_energies.append(energy_from_residual(residual))
        return torch.cat(chunk_energies, dim=0)

    def _register_wi_fov_search(
        self,
        image_width: int,
        image_height: int,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor | None,
        vertex_depth_log_var: torch.Tensor | None,
        init_identity: Optional[torch.Tensor] = None,
        init_x: Optional[torch.Tensor] = None,
        pose_init: bool = True,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_v_canonical, pose_j_canonical = self._calc_canonical_flame(init_identity)

        def evaluate_fov(fov_y: float) -> float:
            camera = self._build_config_camera(image_width, image_height, fov_y)
            camera_view_mat, camera_viewproj_mat = self._single_camera_matrices(camera)
            # Keep the search objective comparable across candidate cameras.

            if pose_init:
                pose_x = self._register_pose(
                    camera_view_mat,
                    camera_viewproj_mat,
                    vertex_coord,
                    vertex_depth,
                    vertex_coord_log_var,
                    vertex_depth_log_var,
                    pose_v_canonical,
                    pose_j_canonical,
                    init_x=init_x,
                )
            else:
                pose_x = init_x

            identity, x = self._register(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_depth_log_var,
                init_identity=init_identity,
                init_x=pose_x,
                num_iters=self.config.fov_registration_iterations,
            )
            return float(
                self._evaluate_energy(
                    camera_view_mat,
                    camera_viewproj_mat,
                    vertex_coord,
                    vertex_depth,
                    vertex_coord_log_var,
                    vertex_depth_log_var,
                    identity,
                    x,
                ).mean().item()
            )

        l = float(self.config.fov_y_min)
        r = float(self.config.fov_y_max)
        phi = 0.5 * (math.sqrt(5.0) - 1.0)

        x1 = r - phi * (r - l)
        x2 = l + phi * (r - l)
        e1 = evaluate_fov(x1)
        e2 = evaluate_fov(x2)

        for iter_id in range(self.config.fov_search_iterations):
            if r - l < self.config.fov_search_epsilon:
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

        best_fov_y = 0.5 * (l + r)
        best_focal = self._fov_y_to_focal(image_height, best_fov_y)
        final_camera = self._build_config_camera(image_width, image_height, best_fov_y)
        final_view_mat, final_viewproj_mat = self._single_camera_matrices(final_camera)

        if pose_init:
            pose_x = self._register_pose(
                final_view_mat,
                final_viewproj_mat,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_depth_log_var,
                pose_v_canonical,
                pose_j_canonical,
                init_x=init_x,
            )
        else:
            pose_x = init_x

        final_identity, final_x = self._register(
            final_view_mat,
            final_viewproj_mat,
            vertex_coord,
            vertex_depth,
            vertex_coord_log_var,
            vertex_depth_log_var,
            init_identity=init_identity,
            init_x=pose_x,
        )
        return best_fov_y, final_identity, final_x, pose_x

    def _register(
        self,
        camera_view_mat: torch.Tensor,
        camera_viewproj_mat: torch.Tensor,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor | None = None,
        vertex_depth_log_var: torch.Tensor | None = None,
        init_identity: Optional[torch.Tensor] = None,
        init_x: Optional[torch.Tensor] = None,
        num_iters: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_views = int(camera_view_mat.shape[-3])

        if vertex_coord.ndim == 2 and vertex_depth.ndim == 2:
            shape = (self.dim_x,)
        elif vertex_coord.ndim == 3 and vertex_depth.ndim == 3:
            if vertex_coord.shape[0] == num_views:
                shape = (self.dim_x,)
            else:
                shape = (vertex_coord.shape[0], self.dim_x)
        elif vertex_coord.ndim == 4 and vertex_depth.ndim == 4:
            shape = (vertex_coord.shape[0], self.dim_x)
        else:
            raise ValueError("Invalid shape of vertex_coord or vertex_depth.")

        if init_identity is None:
            identity = torch.zeros((300,), dtype=torch.float32, device=self.device)
        else:
            identity = init_identity.to(device=self.device, dtype=torch.float32).reshape(300).clone()

        if init_x is None:
            x = torch.zeros(shape, dtype=torch.float32, device=self.device)
        else:
            x = init_x.to(device=self.device, dtype=torch.float32).reshape(shape).clone()

        effective_num_iters = self.config.registration_iterations if num_iters is None else int(num_iters)
        num_batch_frames = x.shape[0] if x.ndim == 2 else 1
        for it in range(effective_num_iters):
            v_canonical, j_canonical = self._calc_canonical_flame(identity)
            damping_lambda = self.config.damping

            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x, v_canonical, j_canonical, self.expr_dirs, self.pose_dirs, self.lbs_weights
            )
            residual, jacobian = self.solver_ops.assemble_x_system(
                camera_view_mat,
                camera_viewproj_mat,
                flame_v,
                flame_j,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_depth_log_var,
                x,
                flame_v_jacobian,
                flame_j_jacobian,
                self.config.correspondence_weight,
                self.config.relative_depth_weight,
                self.config.expression_regularization,
                self.config.pose_regularization,
                self.key_vertex_ids,
            )
            x = x + self.solver_ops.solve_normal_equation(
                self.solver_context,
                jacobian,
                residual,
                damping_lambda,
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
                vertex_depth_log_var,
                identity if init_identity is None else identity - init_identity,
                flame_v_jacobian,
                flame_j_jacobian,
                self.config.correspondence_weight,
                self.config.relative_depth_weight,
                self.config.identity_regularization,
                self.key_vertex_ids,
            )
            identity = identity + self.solver_ops.solve_normal_equation(
                self.solver_context,
                jacobian,
                residual,
                damping_lambda * num_batch_frames,
            )

        return identity, x

    def _register_pose(
        self,
        camera_view_mat: torch.Tensor,
        camera_viewproj_mat: torch.Tensor,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor | None,
        vertex_depth_log_var: torch.Tensor | None,
        v_canonical: torch.Tensor,
        j_canonical: torch.Tensor,
        init_x: Optional[torch.Tensor] = None,
        num_iters: int | None = None,
    ) -> torch.Tensor:
        num_views = int(camera_view_mat.shape[0])
        if vertex_coord.ndim == 2 and vertex_depth.ndim == 2:
            shape = (self.dim_x,)
        elif vertex_coord.ndim == 3 and vertex_depth.ndim == 3:
            if vertex_coord.shape[0] == num_views:
                shape = (self.dim_x,)
            else:
                shape = (vertex_coord.shape[0], self.dim_x)
        elif vertex_coord.ndim == 4 and vertex_depth.ndim == 4:
            shape = (vertex_coord.shape[0], self.dim_x)
        else:
            raise ValueError("Invalid shape of vertex_coord or vertex_depth.")

        if init_x is None:
            x = torch.zeros(shape, dtype=torch.float32, device=self.device)
        else:
            x = init_x.to(device=self.device, dtype=torch.float32).reshape(shape).clone()

        opt_dims = torch.tensor(
            [
                self.pose_start, self.pose_start + 1, self.pose_start + 2,
                self.trans_start, self.trans_start + 1, self.trans_start + 2,
            ],
            dtype=torch.int64,
            device=self.device,
        )
        effective_num_iters = self.config.pose_registration_iterations if num_iters is None else int(num_iters)
        for _ in range(effective_num_iters):
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
                vertex_depth_log_var,
                x,
                flame_v_jacobian,
                flame_j_jacobian,
                alignment_weight=1.0,
                rel_depth_weight=0.0,
                exp_reg_weight=0.0,
                pose_reg_weight=0.0,
                vertex_indices=self.key_vertex_ids,
            )
            delta = self.solver_ops.solve_normal_equation(
                self.solver_context,
                jacobian[..., opt_dims],
                residual,
                self.config.pose_damping,
            )
            x[..., opt_dims] = x[..., opt_dims] + delta
        
        return x

    def _track(
        self,
        camera_view_mat: torch.Tensor,
        camera_viewproj_mat: torch.Tensor,
        vertex_coord: torch.Tensor,
        vertex_depth: torch.Tensor,
        vertex_coord_log_var: torch.Tensor | None,
        vertex_depth_log_var: torch.Tensor | None,
        v_canonical: torch.Tensor,
        j_canonical: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        # Root-pose pre-step: push bulk rotation into root joint first
        root_pose_dims = torch.tensor(
            [self.pose_start, self.pose_start + 1, self.pose_start + 2,
             self.trans_start, self.trans_start + 1, self.trans_start + 2],
            dtype=torch.int64, device=self.device,
        )
        for _ in range(self.config.root_pose_tracking_iterations):
            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x, v_canonical, j_canonical, self.expr_dirs, self.pose_dirs, self.lbs_weights,
            )
            residual, jacobian = self.solver_ops.assemble_x_system(
                camera_view_mat, camera_viewproj_mat,
                flame_v, flame_j, vertex_coord, vertex_depth,
                vertex_coord_log_var, vertex_depth_log_var,
                x, flame_v_jacobian, flame_j_jacobian,
                self.config.correspondence_weight, self.config.relative_depth_weight, 0.0, 0.0,
                self.key_vertex_ids,
            )
            delta = self.solver_ops.solve_normal_equation(
                self.solver_context, jacobian[..., root_pose_dims], residual, self.config.damping,
            )
            x[..., root_pose_dims] = x[..., root_pose_dims] + delta

        # Full optimization with early stopping
        prev_energy = float('inf')
        steps = 0
        for _ in range(self.config.tracking_iterations):
            flame_v, flame_j, flame_v_jacobian, flame_j_jacobian = self.solver_ops.expression_jacobian(
                x, v_canonical, j_canonical, self.expr_dirs, self.pose_dirs, self.lbs_weights,
            )
            residual, jacobian = self.solver_ops.assemble_x_system(
                camera_view_mat, camera_viewproj_mat,
                flame_v, flame_j, vertex_coord, vertex_depth,
                vertex_coord_log_var, vertex_depth_log_var,
                x, flame_v_jacobian, flame_j_jacobian,
                self.config.correspondence_weight, self.config.relative_depth_weight,
                self.config.expression_regularization, self.config.pose_regularization,
                self.key_vertex_ids,
            )

            energy = float(torch.sum(residual ** 2).item())
            if prev_energy - energy < self.config.tracking_early_stop_ratio * prev_energy:
                break
            prev_energy = energy
            steps += 1

            x = x + self.solver_ops.solve_normal_equation(
                self.solver_context, jacobian, residual, self.config.damping,
            )

        return x, steps

    def _track_pass(
        self,
        camera_view_mat: torch.Tensor,
        camera_viewproj_mat: torch.Tensor,
        identity: torch.Tensor,
        init_x: torch.Tensor,
        vertex_coords: torch.Tensor,
        vertex_depths: torch.Tensor,
        vertex_coord_log_vars: torch.Tensor | None,
        vertex_depth_log_vars: torch.Tensor | None,
        desc: str = "track_pass"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_frames = vertex_coords.shape[0]
        xs = torch.zeros((num_frames, self.dim_x), dtype=torch.float32, device=self.device)

        per_frame_init = init_x.ndim == 2
        if not per_frame_init:
            x = init_x.clone()

        v_canonical, j_canonical = self._calc_canonical_flame(identity)
        step_counts = []
        for frame_id in tqdm(range(num_frames), desc=desc, leave=False):
            if per_frame_init:
                x = init_x[frame_id].clone()
            if vertex_coord_log_vars is None or vertex_depth_log_vars is None:
                frame_projected_log_var = vertex_coord_log_vars
                frame_rel_depth_log_var = vertex_depth_log_vars
            else:
                frame_projected_log_var = vertex_coord_log_vars[frame_id]
                frame_rel_depth_log_var = vertex_depth_log_vars[frame_id]
            x, steps = self._track(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coords[frame_id],
                vertex_depths[frame_id],
                frame_projected_log_var,
                frame_rel_depth_log_var,
                v_canonical,
                j_canonical,
                x,
            )
            step_counts.append(steps)
            xs[frame_id] = x
        print(f"  [{desc}] steps: min={min(step_counts)} max={max(step_counts)} mean={np.mean(step_counts):.1f}")

        return xs

    def _select_keyframes(self, identity: torch.Tensor, xs: torch.Tensor, num_keyframes: int) -> torch.Tensor:
        num_frames = xs.shape[0]
        xs_eval = xs.clone()
        xs_eval[:, self.trans_start:self.dim_x] = 0
        v_canonical, j_canonical = self._calc_canonical_flame(identity)
        vertices, _ = self._calc_posed_flame(v_canonical, j_canonical, xs_eval)
        flat_vertices = vertices.flatten(1)

        key_frames = []
        key_frame = int(np.random.randint(num_frames))
        dists = torch.sum((flat_vertices - flat_vertices[key_frame].unsqueeze(0)) ** 2, dim=-1)
        dists[key_frame] = -1.0
        key_frames.append(key_frame)

        for _ in range(num_keyframes - 1):
            key_frame = int(torch.argmax(dists).item())
            dists_to_this = torch.sum((flat_vertices - flat_vertices[key_frame].unsqueeze(0)) ** 2, dim=-1)
            dists = torch.minimum(dists, dists_to_this)
            dists[key_frame] = -1.0
            key_frames.append(key_frame)

        return torch.as_tensor(key_frames, dtype=torch.long, device=self.device)

    def fit_frame(self,
        inference: InferenceOutput,
        camera: Camera | None = None,
        init_identity: np.ndarray | None = None,
    ) -> FittingOutput:
        if inference.vertex_coord.ndim != 2 or inference.vertex_coord.shape[-1] != 2:
            raise ValueError("fit_frame expects inference.vertex_coord with shape [5023, 2].")
        if inference.vertex_depth.ndim != 2 or inference.vertex_depth.shape[-1] != 1:
            raise ValueError("fit_frame expects inference.vertex_depth with shape [5023, 1].")
        if self.config.use_uncertainty_weights:
            if inference.vertex_coord_log_var.ndim != 2 or inference.vertex_coord_log_var.shape[-1] != 1:
                raise ValueError("fit_frame expects inference.vertex_coord_log_var with shape [5023, 1].")
            if inference.vertex_depth_log_var.ndim != 2 or inference.vertex_depth_log_var.shape[-1] != 1:
                raise ValueError("fit_frame expects inference.vertex_depth_log_var with shape [5023, 1].")

        vertex_coord = torch.from_numpy(inference.vertex_coord).to(device=self.device, dtype=torch.float32)
        vertex_depth = torch.from_numpy(inference.vertex_depth).to(device=self.device, dtype=torch.float32)
        vertex_coord[..., 0] /= float(inference.image_width)
        vertex_coord[..., 1] /= float(inference.image_height)
        if self.config.use_uncertainty_weights:
            vertex_coord_log_var: torch.Tensor | None = torch.from_numpy(inference.vertex_coord_log_var).to(device=self.device, dtype=torch.float32)
            vertex_depth_log_var: torch.Tensor | None = torch.from_numpy(inference.vertex_depth_log_var).to(device=self.device, dtype=torch.float32)
        else:
            vertex_coord_log_var = None
            vertex_depth_log_var = None
        vertex_coord_log_var = self._apply_eye_region_boost(vertex_coord_log_var)

        if init_identity is None:
            pose_identity = torch.zeros((300,), dtype=torch.float32, device=self.device)
        else:
            pose_identity = torch.from_numpy(init_identity).to(dtype=torch.float32, device=self.device)

        camera_fov_y = None
        if camera is None:
            camera_fov_y, identity, x, pose_x = self._register_wi_fov_search(
                inference.image_width,
                inference.image_height,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_depth_log_var,
                init_identity=pose_identity,
            )
        else:
            camera_view_mat, camera_viewproj_mat = self._single_camera_matrices(camera)
            pose_v_canonical, pose_j_canonical = self._calc_canonical_flame(pose_identity)
            pose_x = self._register_pose(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_depth_log_var,
                pose_v_canonical,
                pose_j_canonical,
            )
            identity, x = self._register(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coord,
                vertex_depth,
                vertex_coord_log_var,
                vertex_depth_log_var,
                init_identity=pose_identity,
                init_x=pose_x,
            )

        return FittingOutput(identity=identity, x=x, pose_x=pose_x, camera_fov_y=camera_fov_y)

    def fit_frame_mv(self, inferences: list[InferenceOutput], cameras: list[Camera]) -> FittingOutput:
        if len(inferences) == 0:
            raise ValueError("fit_frame_mv expects at least one InferenceOutput.")
        if len(inferences) != len(cameras):
            raise ValueError("fit_frame_mv expects the same number of inferences and cameras.")
        if any(camera is None for camera in cameras):
            raise ValueError("fit_frame_mv requires an explicit camera for each view.")

        vertex_coords = []
        vertex_depths = []
        vertex_coord_log_vars = []
        vertex_depth_log_vars = []
        for inference in inferences:
            if inference.vertex_coord.ndim != 2 or inference.vertex_coord.shape[-1] != 2:
                raise ValueError("fit_frame_mv expects each inference.vertex_coord with shape [5023, 2].")
            if inference.vertex_depth.ndim != 2 or inference.vertex_depth.shape[-1] != 1:
                raise ValueError("fit_frame_mv expects each inference.vertex_depth with shape [5023, 1].")
            if self.config.use_uncertainty_weights:
                if inference.vertex_coord_log_var.ndim != 2 or inference.vertex_coord_log_var.shape[-1] != 1:
                    raise ValueError("fit_frame_mv expects each inference.vertex_coord_log_var with shape [5023, 1].")
                if inference.vertex_depth_log_var.ndim != 2 or inference.vertex_depth_log_var.shape[-1] != 1:
                    raise ValueError("fit_frame_mv expects each inference.vertex_depth_log_var with shape [5023, 1].")

            vertex_coord = torch.from_numpy(inference.vertex_coord).to(device=self.device, dtype=torch.float32)
            vertex_depth = torch.from_numpy(inference.vertex_depth).to(device=self.device, dtype=torch.float32)
            vertex_coord[..., 0] /= float(inference.image_width)
            vertex_coord[..., 1] /= float(inference.image_height)
            vertex_coords.append(vertex_coord)
            vertex_depths.append(vertex_depth)
            if self.config.use_uncertainty_weights:
                vertex_coord_log_vars.append(
                    torch.from_numpy(inference.vertex_coord_log_var).to(device=self.device, dtype=torch.float32)
                )
                vertex_depth_log_vars.append(
                    torch.from_numpy(inference.vertex_depth_log_var).to(device=self.device, dtype=torch.float32)
                )

        vertex_coords_tensor = torch.stack(vertex_coords, dim=0)  # [C, V, 2]
        vertex_depths_tensor = torch.stack(vertex_depths, dim=0)  # [C, V, 1]
        if self.config.use_uncertainty_weights:
            vertex_coord_log_vars_tensor: torch.Tensor | None = torch.stack(vertex_coord_log_vars, dim=0)  # [C, V, 1]
            vertex_depth_log_vars_tensor: torch.Tensor | None = torch.stack(vertex_depth_log_vars, dim=0)  # [C, V, 1]
        else:
            vertex_coord_log_vars_tensor = None
            vertex_depth_log_vars_tensor = None
        vertex_coord_log_vars_tensor = self._apply_eye_region_boost(vertex_coord_log_vars_tensor)
        camera_view_mat, camera_viewproj_mat = self._stack_camera_matrices(cameras)
        pose_identity = torch.zeros((300,), dtype=torch.float32, device=self.device)
        pose_v_canonical, pose_j_canonical = self._calc_canonical_flame(pose_identity)
        pose_x = self._register_pose(
            camera_view_mat,
            camera_viewproj_mat,
            vertex_coords_tensor,
            vertex_depths_tensor,
            vertex_coord_log_vars_tensor,
            vertex_depth_log_vars_tensor,
            pose_v_canonical,
            pose_j_canonical,
        )
        identity, x = self._register(
            camera_view_mat,
            camera_viewproj_mat,
            vertex_coords_tensor,
            vertex_depths_tensor,
            vertex_coord_log_vars_tensor,
            vertex_depth_log_vars_tensor,
            init_x=pose_x,
        )

        return FittingOutput(identity=identity, x=x)

    def fit_sequence(self,
        inference: InferenceOutput,
        camera: Camera | None = None,
        init_identity: np.ndarray | None = None,
    ) -> FittingOutput:
        if inference.vertex_coord.ndim != 3 or inference.vertex_coord.shape[-1] != 2:
            raise ValueError("fit_sequence expects inference.vertex_coord with shape [T, 5023, 2].")
        if inference.vertex_depth.ndim != 3 or inference.vertex_depth.shape[-1] != 1:
            raise ValueError("fit_sequence expects inference.vertex_depth with shape [T, 5023, 1].")
        if self.config.use_uncertainty_weights:
            if inference.vertex_coord_log_var.ndim != 3 or inference.vertex_coord_log_var.shape[-1] != 1:
                raise ValueError("fit_sequence expects inference.vertex_coord_log_var with shape [T, 5023, 1].")
            if inference.vertex_depth_log_var.ndim != 3 or inference.vertex_depth_log_var.shape[-1] != 1:
                raise ValueError("fit_sequence expects inference.vertex_depth_log_var with shape [T, 5023, 1].")

        vertex_coords = torch.from_numpy(inference.vertex_coord).to(device=self.device, dtype=torch.float32)
        vertex_depths = torch.from_numpy(inference.vertex_depth).to(device=self.device, dtype=torch.float32)
        vertex_coords[..., 0] /= float(inference.image_width)
        vertex_coords[..., 1] /= float(inference.image_height)
        if self.config.use_uncertainty_weights:
            vertex_coord_log_vars: torch.Tensor | None = torch.from_numpy(inference.vertex_coord_log_var).to(device=self.device, dtype=torch.float32)
            vertex_depth_log_vars: torch.Tensor | None = torch.from_numpy(inference.vertex_depth_log_var).to(device=self.device, dtype=torch.float32)
        else:
            vertex_coord_log_vars = None
            vertex_depth_log_vars = None
        vertex_coord_log_vars = self._apply_eye_region_boost(vertex_coord_log_vars)
        if vertex_coord_log_vars is not None:
            frame0_coord_log_var: torch.Tensor | None = vertex_coord_log_vars[0]
            frame0_depth_log_var: torch.Tensor | None = vertex_depth_log_vars[0]
        else:
            frame0_coord_log_var = None
            frame0_depth_log_var = None

        fit_camera = camera is None

        if init_identity is None:
            init_identity = torch.zeros((300,), dtype=torch.float32, device=self.device)
        else:
            init_identity = torch.from_numpy(init_identity).to(dtype=torch.float32, device=self.device)
        init_v_canonical, init_j_canonical = self._calc_canonical_flame(init_identity)

        print("[GN] register first frame")
        camera_fov_y = None
        if fit_camera:
            camera_fov_y, identity, x0, pose_x0 = self._register_wi_fov_search(
                inference.image_width,
                inference.image_height,
                vertex_coords[0],
                vertex_depths[0],
                frame0_coord_log_var,
                frame0_depth_log_var,
                init_identity=init_identity,
            )
            print(f"[GN] optimized camera fov_y after first frame register: {camera_fov_y:.6f} deg")
            camera = self._build_config_camera(inference.image_width, inference.image_height, camera_fov_y)
            camera_view_mat, camera_viewproj_mat = self._single_camera_matrices(camera)
        else:
            camera_view_mat, camera_viewproj_mat = self._single_camera_matrices(camera)
            pose_x0 = self._register_pose(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coords[0],
                vertex_depths[0],
                frame0_coord_log_var,
                frame0_depth_log_var,
                init_v_canonical,
                init_j_canonical,
            )
            identity, x0 = self._register(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coords[0],
                vertex_depths[0],
                frame0_coord_log_var,
                frame0_depth_log_var,
                init_identity=init_identity,
                init_x=pose_x0,
            )
        xs = self._track_pass(
            camera_view_mat,
            camera_viewproj_mat,
            identity,
            x0,
            vertex_coords,
            vertex_depths,
            vertex_coord_log_vars,
            vertex_depth_log_vars,
            desc="[GN] track pass 0"
        )

        num_frames = xs.shape[0]
        for round_id in range(self.config.identity_refinement_rounds):
            print(f"[GN] refine round {round_id + 1}/{self.config.identity_refinement_rounds}")
            
            # select keyframes
            if num_frames > self.config.num_keyframes:
                key_frames = self._select_keyframes(identity, xs, self.config.num_keyframes)
                xs_key = xs[key_frames]
                vertex_coords_key = vertex_coords[key_frames]
                vertex_depths_key = vertex_depths[key_frames]
                if vertex_coord_log_vars is None or vertex_depth_log_vars is None:
                    vertex_coord_log_vars_key = None
                    vertex_depth_log_vars_key = None
                else:
                    vertex_coord_log_vars_key = vertex_coord_log_vars[key_frames]
                    vertex_depth_log_vars_key = vertex_depth_log_vars[key_frames]
            else:
                xs_key = xs
                vertex_coords_key = vertex_coords
                vertex_depths_key = vertex_depths
                vertex_coord_log_vars_key = vertex_coord_log_vars
                vertex_depth_log_vars_key = vertex_depth_log_vars
            
            identity, _ = self._register(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coords_key,
                vertex_depths_key,
                vertex_coord_log_vars_key,
                vertex_depth_log_vars_key,
                init_identity=identity,
                init_x=xs_key
            )
            
            # tracking pass
            xs = self._track_pass(
                camera_view_mat,
                camera_viewproj_mat,
                identity,
                xs,
                vertex_coords,
                vertex_depths,
                vertex_coord_log_vars,
                vertex_depth_log_vars,
                desc=f"[GN] track pass {round_id + 1}"
            )

        return FittingOutput(identity=identity, x=xs, camera_fov_y=camera_fov_y)

    def fit_sequence_mv(self, inferences: list[InferenceOutput], cameras: list[Camera]) -> FittingOutput:
        if len(inferences) == 0:
            raise ValueError("fit_sequence_mv expects at least one InferenceOutput.")
        if len(inferences) != len(cameras):
            raise ValueError("fit_sequence_mv expects the same number of inferences and cameras.")
        if any(camera is None for camera in cameras):
            raise ValueError("fit_sequence_mv requires an explicit camera for each view.")

        first_inference = inferences[0]
        if first_inference.vertex_coord.ndim != 3 or first_inference.vertex_coord.shape[-1] != 2:
            raise ValueError("fit_sequence_mv expects each inference.vertex_coord with shape [T, 5023, 2].")
        if first_inference.vertex_depth.ndim != 3 or first_inference.vertex_depth.shape[-1] != 1:
            raise ValueError("fit_sequence_mv expects each inference.vertex_depth with shape [T, 5023, 1].")
        if self.config.use_uncertainty_weights:
            if first_inference.vertex_coord_log_var.ndim != 3 or first_inference.vertex_coord_log_var.shape[-1] != 1:
                raise ValueError("fit_sequence_mv expects each inference.vertex_coord_log_var with shape [T, 5023, 1].")
            if first_inference.vertex_depth_log_var.ndim != 3 or first_inference.vertex_depth_log_var.shape[-1] != 1:
                raise ValueError("fit_sequence_mv expects each inference.vertex_depth_log_var with shape [T, 5023, 1].")

        num_views = len(inferences)
        num_frames = first_inference.vertex_coord.shape[0]
        num_vertices = first_inference.vertex_coord.shape[1]
        vertex_coords_tensor = torch.empty((num_frames, num_views, num_vertices, 2), dtype=torch.float32, device=self.device)
        vertex_depths_tensor = torch.empty((num_frames, num_views, num_vertices, 1), dtype=torch.float32, device=self.device)
        if self.config.use_uncertainty_weights:
            vertex_coord_log_vars_tensor: torch.Tensor | None = torch.empty(
                (num_frames, num_views, num_vertices, 1), dtype=torch.float32, device=self.device
            )
            vertex_depth_log_vars_tensor: torch.Tensor | None = torch.empty(
                (num_frames, num_views, num_vertices, 1), dtype=torch.float32, device=self.device
            )
        else:
            vertex_coord_log_vars_tensor = None
            vertex_depth_log_vars_tensor = None

        for view_id, inference in enumerate(inferences):
            if inference.vertex_coord.ndim != 3 or inference.vertex_coord.shape[-1] != 2:
                raise ValueError("fit_sequence_mv expects each inference.vertex_coord with shape [T, 5023, 2].")
            if inference.vertex_depth.ndim != 3 or inference.vertex_depth.shape[-1] != 1:
                raise ValueError("fit_sequence_mv expects each inference.vertex_depth with shape [T, 5023, 1].")
            if self.config.use_uncertainty_weights:
                if inference.vertex_coord_log_var.ndim != 3 or inference.vertex_coord_log_var.shape[-1] != 1:
                    raise ValueError("fit_sequence_mv expects each inference.vertex_coord_log_var with shape [T, 5023, 1].")
                if inference.vertex_depth_log_var.ndim != 3 or inference.vertex_depth_log_var.shape[-1] != 1:
                    raise ValueError("fit_sequence_mv expects each inference.vertex_depth_log_var with shape [T, 5023, 1].")
            if inference.vertex_coord.shape[0] != num_frames:
                raise ValueError("fit_sequence_mv expects all views to share the same frame count.")

            vertex_coords = torch.from_numpy(inference.vertex_coord).to(device=self.device, dtype=torch.float32)
            vertex_depths = torch.from_numpy(inference.vertex_depth).to(device=self.device, dtype=torch.float32)
            vertex_coords[..., 0] /= float(inference.image_width)
            vertex_coords[..., 1] /= float(inference.image_height)
            vertex_coords_tensor[:, view_id] = vertex_coords
            vertex_depths_tensor[:, view_id] = vertex_depths
            if self.config.use_uncertainty_weights:
                vertex_coord_log_vars_tensor[:, view_id] = torch.from_numpy(inference.vertex_coord_log_var).to(
                    device=self.device, dtype=torch.float32
                )
                vertex_depth_log_vars_tensor[:, view_id] = torch.from_numpy(inference.vertex_depth_log_var).to(
                    device=self.device, dtype=torch.float32
                )

        vertex_coord_log_vars_tensor = self._apply_eye_region_boost(vertex_coord_log_vars_tensor)
        if self.config.use_uncertainty_weights:
            frame0_coord_log_var = vertex_coord_log_vars_tensor[0]
            frame0_depth_log_var = vertex_depth_log_vars_tensor[0]
        else:
            frame0_coord_log_var = None
            frame0_depth_log_var = None
        camera_view_mat, camera_viewproj_mat = self._stack_camera_matrices(cameras)

        print("[GN] register first frame (multi-view)")
        pose_identity = torch.zeros((300,), dtype=torch.float32, device=self.device)
        pose_v_canonical, pose_j_canonical = self._calc_canonical_flame(pose_identity)
        pose_x0 = self._register_pose(
            camera_view_mat,
            camera_viewproj_mat,
            vertex_coords_tensor[0],
            vertex_depths_tensor[0],
            frame0_coord_log_var,
            frame0_depth_log_var,
            pose_v_canonical,
            pose_j_canonical,
        )
        self._check_finite(pose_x0, "after _register_pose: pose_x0")
        identity, x0 = self._register(
            camera_view_mat,
            camera_viewproj_mat,
            vertex_coords_tensor[0],
            vertex_depths_tensor[0],
            frame0_coord_log_var,
            frame0_depth_log_var,
            init_x=pose_x0,
        )
        self._check_finite(identity, "after first _register: identity")
        self._check_finite(x0, "after first _register: x0")
        xs = self._track_pass(
            camera_view_mat,
            camera_viewproj_mat,
            identity,
            x0,
            vertex_coords_tensor,
            vertex_depths_tensor,
            vertex_coord_log_vars_tensor,
            vertex_depth_log_vars_tensor,
            desc="[GN] track pass 0 (multi-view)"
        )
        self._check_finite(xs, "after track pass 0: xs")

        num_frames = xs.shape[0]
        for round_id in range(self.config.identity_refinement_rounds):
            print(f"[GN] refine round {round_id + 1}/{self.config.identity_refinement_rounds} (multi-view)")

            # select key frames
            if num_frames > self.config.num_keyframes:
                key_frames = self._select_keyframes(identity, xs, self.config.num_keyframes)
                xs_key = xs[key_frames]
                vertex_coords_key = vertex_coords_tensor[key_frames]
                vertex_depths_key = vertex_depths_tensor[key_frames]
                if vertex_coord_log_vars_tensor is None or vertex_depth_log_vars_tensor is None:
                    vertex_coord_log_vars_key = None
                    vertex_depth_log_vars_key = None
                else:
                    vertex_coord_log_vars_key = vertex_coord_log_vars_tensor[key_frames]
                    vertex_depth_log_vars_key = vertex_depth_log_vars_tensor[key_frames]
            else:
                xs_key = xs
                vertex_coords_key = vertex_coords_tensor
                vertex_depths_key = vertex_depths_tensor
                vertex_coord_log_vars_key = vertex_coord_log_vars_tensor
                vertex_depth_log_vars_key = vertex_depth_log_vars_tensor
            self._check_finite(xs_key, f"refine {round_id + 1}: xs_key (after keyframe select)")

            # register pass
            identity, _ = self._register(
                camera_view_mat,
                camera_viewproj_mat,
                vertex_coords_key,
                vertex_depths_key,
                vertex_coord_log_vars_key,
                vertex_depth_log_vars_key,
                init_identity=identity,
                init_x=xs_key,
            )
            self._check_finite(identity, f"refine {round_id + 1}: identity (after _register)")

            # tracking pass
            xs = self._track_pass(
                camera_view_mat,
                camera_viewproj_mat,
                identity,
                xs,
                vertex_coords_tensor,
                vertex_depths_tensor,
                vertex_coord_log_vars_tensor,
                vertex_depth_log_vars_tensor,
                desc=f"[GN] track pass {round_id + 1} (multi-view)"
            )
            self._check_finite(xs, f"refine {round_id + 1}: xs (after track pass)")

        return FittingOutput(identity=identity, x=xs)

    # ---- Robust multi-view tracking (adaptive per-frame camera selection) ----

    def _track_pass_robust(
        self,
        all_cameras: list[Camera],
        identity: torch.Tensor,
        init_x: torch.Tensor,
        per_view_coords: list[torch.Tensor],
        per_view_depths: list[torch.Tensor],
        per_view_coord_log_vars: list[torch.Tensor] | None,
        per_view_depth_log_vars: list[torch.Tensor] | None,
        per_frame_views: list[list[int]],
        frame_reliable: torch.Tensor,
        desc: str = "track_pass_robust",
    ) -> torch.Tensor:
        num_frames = len(per_frame_views)
        xs = torch.zeros((num_frames, self.dim_x), dtype=torch.float32, device=self.device)

        per_frame_init = init_x.ndim == 2
        if not per_frame_init:
            x = init_x.clone()

        v_canonical, j_canonical = self._calc_canonical_flame(identity)
        step_counts = []
        for t in tqdm(range(num_frames), desc=desc, leave=False):
            if per_frame_init:
                x = init_x[t].clone()

            if not frame_reliable[t]:
                xs[t] = x
                continue

            views = per_frame_views[t]
            cam_view, cam_proj = self._stack_camera_matrices([all_cameras[v] for v in views])
            frame_coords = torch.stack([per_view_coords[v][t] for v in views])
            frame_depths = torch.stack([per_view_depths[v][t] for v in views])
            if per_view_coord_log_vars is not None:
                frame_coord_lv = torch.stack([per_view_coord_log_vars[v][t] for v in views])
                frame_depth_lv = torch.stack([per_view_depth_log_vars[v][t] for v in views])
            else:
                frame_coord_lv = None
                frame_depth_lv = None

            x, steps = self._track(
                cam_view, cam_proj,
                frame_coords, frame_depths,
                frame_coord_lv, frame_depth_lv,
                v_canonical, j_canonical, x,
            )
            step_counts.append(steps)
            xs[t] = x

        print(f"  [{desc}] steps: min={min(step_counts)} max={max(step_counts)} mean={np.mean(step_counts):.1f}")
        return xs

    def _gather_register_data(
        self,
        keyframe_indices: torch.Tensor,
        per_frame_views: list[list[int]],
        all_cameras: list[Camera],
        per_view_coords: list[torch.Tensor],
        per_view_depths: list[torch.Tensor],
        per_view_coord_log_vars: list[torch.Tensor] | None,
        per_view_depth_log_vars: list[torch.Tensor] | None,
        num_vertices: int,
    ):
        key_ts = keyframe_indices.cpu().tolist()
        n_keys = len(key_ts)

        per_key_views = [per_frame_views[t] for t in key_ts]
        c_max = max(len(v) for v in per_key_views)

        cam_views_list = []
        cam_projs_list = []
        for ki in range(n_keys):
            views = per_key_views[ki]
            cv, cp = self._stack_camera_matrices([all_cameras[v] for v in views])
            if len(views) < c_max:
                pad = c_max - len(views)
                cv = torch.cat([cv, cv[:1].expand(pad, -1, -1)], dim=0)
                cp = torch.cat([cp, cp[:1].expand(pad, -1, -1)], dim=0)
            cam_views_list.append(cv)
            cam_projs_list.append(cp)
        cam_view = torch.stack(cam_views_list)
        cam_proj = torch.stack(cam_projs_list)

        coords = torch.full((n_keys, c_max, num_vertices, 2), 0.5,
                            dtype=torch.float32, device=self.device)
        depths = torch.zeros((n_keys, c_max, num_vertices, 1),
                             dtype=torch.float32, device=self.device)
        coord_lvs = torch.full((n_keys, c_max, num_vertices, 1), INVISIBLE_LOG_VAR,
                               dtype=torch.float32, device=self.device)
        depth_lvs = torch.full((n_keys, c_max, num_vertices, 1), INVISIBLE_LOG_VAR,
                               dtype=torch.float32, device=self.device)

        for ki, t in enumerate(key_ts):
            for ci, v in enumerate(per_key_views[ki]):
                coords[ki, ci] = per_view_coords[v][t]
                depths[ki, ci] = per_view_depths[v][t]
                if per_view_coord_log_vars is not None:
                    coord_lvs[ki, ci] = per_view_coord_log_vars[v][t]
                    depth_lvs[ki, ci] = per_view_depth_log_vars[v][t]

        if per_view_coord_log_vars is None:
            return cam_view, cam_proj, coords, depths, None, None
        return cam_view, cam_proj, coords, depths, coord_lvs, depth_lvs

    def fit_sequence_mv_robust(
        self,
        inferences: list[InferenceOutput],
        cameras: list[Camera],
        init_x: Optional[torch.Tensor] = None,
        n_max_views: int = 16,
        n_min_views: int = 3,
    ) -> FittingOutput:
        num_all_views = len(inferences)
        num_frames = inferences[0].vertex_coord.shape[0]
        num_vertices = inferences[0].vertex_coord.shape[1]

        # --- Per-view data to GPU tensors ---
        per_view_coords: list[torch.Tensor] = []
        per_view_depths: list[torch.Tensor] = []
        per_view_coord_log_vars: list[torch.Tensor] | None = []
        per_view_depth_log_vars: list[torch.Tensor] | None = []

        for inference in inferences:
            vc = torch.from_numpy(np.asarray(inference.vertex_coord)).to(
                device=self.device, dtype=torch.float32)
            vd = torch.from_numpy(np.asarray(inference.vertex_depth)).to(
                device=self.device, dtype=torch.float32)
            vc[..., 0] /= float(inference.image_width)
            vc[..., 1] /= float(inference.image_height)
            per_view_coords.append(vc)
            per_view_depths.append(vd)
            if self.config.use_uncertainty_weights:
                per_view_coord_log_vars.append(
                    torch.from_numpy(np.asarray(inference.vertex_coord_log_var)).to(
                        device=self.device, dtype=torch.float32))
                per_view_depth_log_vars.append(
                    torch.from_numpy(np.asarray(inference.vertex_depth_log_var)).to(
                        device=self.device, dtype=torch.float32))

        if self.config.use_uncertainty_weights:
            per_view_coord_log_vars = [self._apply_eye_region_boost(lv) for lv in per_view_coord_log_vars]
        else:
            per_view_coord_log_vars = None
            per_view_depth_log_vars = None

        # --- Per-frame view selection ---
        frame_valid_matrix = np.ones((num_frames, num_all_views), dtype=bool)
        for v, inf in enumerate(inferences):
            if inf.frame_valid is not None:
                frame_valid_matrix[:, v] = np.asarray(inf.frame_valid)

        per_frame_views: list[list[int]] = []
        frame_reliable = torch.ones(num_frames, dtype=torch.bool, device=self.device)

        for t in range(num_frames):
            valid = [v for v in range(num_all_views) if frame_valid_matrix[t, v]]
            if len(valid) < n_min_views:
                print(f"[GN][warn] frame {t}: only {len(valid)} valid views (< {n_min_views}), marked unreliable")
                frame_reliable[t] = False
                per_frame_views.append([])
            elif len(valid) > n_max_views:
                per_frame_views.append(valid[:n_max_views])
            else:
                per_frame_views.append(valid)

        n_reliable = int(frame_reliable.sum().item())
        view_counts = [len(v) for v in per_frame_views]
        print(f"[GN] {n_reliable}/{num_frames} reliable frames, "
              f"views per frame: mean={np.mean(view_counts):.1f} min={np.min(view_counts)} max={np.max(view_counts)} "
              f"(n_max_views={n_max_views}, n_min_views={n_min_views})")

        # --- Register first reliable frame ---
        first_t = int(torch.argmax(frame_reliable.int()).item())
        first_views = per_frame_views[first_t]
        first_cam_view, first_cam_proj = self._stack_camera_matrices(
            [cameras[v] for v in first_views])
        first_coords = torch.stack([per_view_coords[v][first_t] for v in first_views])
        first_depths = torch.stack([per_view_depths[v][first_t] for v in first_views])
        if per_view_coord_log_vars is not None:
            first_coord_lv = torch.stack([per_view_coord_log_vars[v][first_t] for v in first_views])
            first_depth_lv = torch.stack([per_view_depth_log_vars[v][first_t] for v in first_views])
        else:
            first_coord_lv = None
            first_depth_lv = None

        print(f"[GN] register frame {first_t} with {len(first_views)} views (robust)")
        pose_identity = torch.zeros((300,), dtype=torch.float32, device=self.device)
        pose_v, pose_j = self._calc_canonical_flame(pose_identity)
        pose_x0 = self._register_pose(
            first_cam_view, first_cam_proj,
            first_coords, first_depths,
            first_coord_lv, first_depth_lv,
            pose_v, pose_j,
            init_x=init_x,
        )
        identity, x0 = self._register(
            first_cam_view, first_cam_proj,
            first_coords, first_depths,
            first_coord_lv, first_depth_lv,
            init_x=pose_x0,
        )

        # --- Track pass 0 ---
        xs = self._track_pass_robust(
            cameras, identity, x0,
            per_view_coords, per_view_depths,
            per_view_coord_log_vars, per_view_depth_log_vars,
            per_frame_views, frame_reliable,
            desc="[GN] track pass 0 (robust)",
        )

        # --- Refine rounds ---
        reliable_indices = torch.where(frame_reliable)[0]
        register_keyframes: list[list[int]] = []
        for round_id in range(self.config.identity_refinement_rounds):
            print(f"[GN] refine round {round_id + 1}/{self.config.identity_refinement_rounds} (robust)")

            xs_reliable = xs[reliable_indices]
            n_keys = min(self.config.num_keyframes, len(reliable_indices))
            if len(reliable_indices) > n_keys:
                key_in_reliable = self._select_keyframes(identity, xs_reliable, n_keys)
                key_frames = reliable_indices[key_in_reliable]
            else:
                key_frames = reliable_indices

            register_keyframes.append(key_frames.cpu().tolist())

            cam_view, cam_proj, reg_coords, reg_depths, reg_coord_lv, reg_depth_lv = \
                self._gather_register_data(
                    key_frames, per_frame_views, cameras,
                    per_view_coords, per_view_depths,
                    per_view_coord_log_vars, per_view_depth_log_vars,
                    num_vertices,
                )
            xs_key = xs[key_frames]
            print(f"[GN]   {len(key_frames)} keyframes, {cam_view.shape[1]} cameras for register pass")

            identity, _ = self._register(
                cam_view, cam_proj,
                reg_coords, reg_depths,
                reg_coord_lv, reg_depth_lv,
                init_identity=identity,
                init_x=xs_key,
            )

            xs = self._track_pass_robust(
                cameras, identity, xs,
                per_view_coords, per_view_depths,
                per_view_coord_log_vars, per_view_depth_log_vars,
                per_frame_views, frame_reliable,
                desc=f"[GN] track pass {round_id + 1} (robust)",
            )

        return FittingOutput(
            identity=identity,
            x=xs,
            per_frame_views=per_frame_views,
            frame_reliable=frame_reliable.cpu(),
            register_keyframes=register_keyframes,
        )

    def decode(self, output: FittingOutput) -> tuple[torch.Tensor, torch.Tensor]:
        identity = output.identity.to(device=self.device, dtype=torch.float32).reshape(300)
        x = output.x.to(device=self.device, dtype=torch.float32)
        squeeze_batch = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True
        if x.ndim != 2 or x.shape[-1] != self.dim_x:
            raise ValueError(
                f"decode expects output.x with shape [{self.dim_x}] or [T, {self.dim_x}]."
            )

        v_canonical, j_canonical = self._calc_canonical_flame(identity)
        vertices, joints = self._calc_posed_flame(v_canonical, j_canonical, x)
        if squeeze_batch:
            return vertices[0], joints[0]
        return vertices, joints

