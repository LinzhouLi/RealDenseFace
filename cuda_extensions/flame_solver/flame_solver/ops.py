from __future__ import annotations

import torch

from . import cuda_ext


class FlameSolverOps:
    def __init__(self, num_expressions: int = 100, extension=None):
        self.num_expressions = int(num_expressions)
        self.extension = cuda_ext if extension is None else extension

    @property
    def available(self) -> bool:
        return True

    @property
    def dim_x(self) -> int:
        return self.num_expressions + 18

    def calc_canonical(self, *args, **kwargs):
        return self.extension.flame_calc_canonical(*args, **kwargs)

    def expression_jacobian(self, *args):
        return self.extension.flame_expression_jacobian(*args, self.num_expressions)

    def forward_vertices(self, *args):
        return self.extension.flame_expression_forward(*args, self.num_expressions)

    def identity_jacobian(self, *args):
        return self.extension.flame_identity_jacobian(*args, self.num_expressions)

    def _normalize_camera(self, view_mat: torch.Tensor, viewproj_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if view_mat.ndim == 2:
            view_mat = view_mat.unsqueeze(0)
        if viewproj_mat.ndim == 2:
            viewproj_mat = viewproj_mat.unsqueeze(0)
        if view_mat.ndim not in (3, 4) or viewproj_mat.ndim not in (3, 4):
            raise ValueError("Camera matrices must be [4,4], [C,4,4], or [T,C,4,4].")
        if view_mat.ndim != viewproj_mat.ndim:
            raise ValueError("view_mat and viewproj_mat must have the same number of dimensions.")
        if tuple(view_mat.shape[-2:]) != (4, 4) or tuple(viewproj_mat.shape[-2:]) != (4, 4):
            raise ValueError("Camera matrices must end with [4,4].")
        if view_mat.shape != viewproj_mat.shape:
            raise ValueError("view_mat and viewproj_mat must have the same shape.")
        return view_mat, viewproj_mat

    def _normalize_gt_observations(
        self,
        gt_projected_vertices: torch.Tensor,
        gt_vertex_rel_depths: torch.Tensor,
        num_frames: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gt_projected_vertices.ndim == 2:
            gt_projected_vertices = gt_projected_vertices.unsqueeze(0).unsqueeze(0)
        elif gt_projected_vertices.ndim == 3:
            if gt_projected_vertices.shape[0] == num_frames:
                gt_projected_vertices = gt_projected_vertices.unsqueeze(1)
            elif num_frames == 1 and gt_projected_vertices.shape[0] == num_views:
                gt_projected_vertices = gt_projected_vertices.unsqueeze(0)
            else:
                raise ValueError("Unsupported gt_projected_vertices shape.")
        elif gt_projected_vertices.ndim != 4:
            raise ValueError("gt_projected_vertices must be [V,2], [T,V,2], [C,V,2], or [T,C,V,2].")
        if gt_projected_vertices.shape[0] != num_frames or gt_projected_vertices.shape[1] != num_views:
            raise ValueError("gt_projected_vertices must match the number of frames and views.")
        if gt_projected_vertices.shape[-1] != 2:
            raise ValueError("gt_projected_vertices must end with size 2.")

        num_vertices = gt_projected_vertices.shape[2]

        if gt_vertex_rel_depths.ndim == 1:
            gt_vertex_rel_depths = gt_vertex_rel_depths.view(1, 1, -1, 1)
        elif gt_vertex_rel_depths.ndim == 2:
            if gt_vertex_rel_depths.shape[1] == 1:
                gt_vertex_rel_depths = gt_vertex_rel_depths.unsqueeze(0).unsqueeze(0)
            elif gt_vertex_rel_depths.shape[0] == num_frames and gt_vertex_rel_depths.shape[1] == num_vertices:
                gt_vertex_rel_depths = gt_vertex_rel_depths.unsqueeze(1).unsqueeze(-1)
            elif num_frames == 1 and gt_vertex_rel_depths.shape[0] == num_views and gt_vertex_rel_depths.shape[1] == num_vertices:
                gt_vertex_rel_depths = gt_vertex_rel_depths.unsqueeze(0).unsqueeze(-1)
            else:
                raise ValueError("Unsupported gt_vertex_rel_depths shape.")
        elif gt_vertex_rel_depths.ndim == 3:
            if gt_vertex_rel_depths.shape[-1] == 1:
                if gt_vertex_rel_depths.shape[0] == num_frames and gt_vertex_rel_depths.shape[1] == num_vertices:
                    gt_vertex_rel_depths = gt_vertex_rel_depths.unsqueeze(1)
                elif num_frames == 1 and gt_vertex_rel_depths.shape[0] == num_views and gt_vertex_rel_depths.shape[1] == num_vertices:
                    gt_vertex_rel_depths = gt_vertex_rel_depths.unsqueeze(0)
                else:
                    raise ValueError("Unsupported gt_vertex_rel_depths shape.")
            else:
                if (
                    gt_vertex_rel_depths.shape[0] == num_frames
                    and gt_vertex_rel_depths.shape[1] == num_views
                    and gt_vertex_rel_depths.shape[2] == num_vertices
                ):
                    gt_vertex_rel_depths = gt_vertex_rel_depths.unsqueeze(-1)
                else:
                    raise ValueError("Unsupported gt_vertex_rel_depths shape.")
        elif gt_vertex_rel_depths.ndim != 4:
            raise ValueError("gt_vertex_rel_depths must be [V], [V,1], [T,V], [T,V,1], [C,V], [C,V,1], [T,C,V], or [T,C,V,1].")

        if gt_vertex_rel_depths.shape[-1] != 1:
            raise ValueError("gt_vertex_rel_depths must end with size 1.")
        if tuple(gt_vertex_rel_depths.shape[:3]) != tuple(gt_projected_vertices.shape[:3]):
            raise ValueError("gt_vertex_rel_depths must match gt_projected_vertices in [T,C,V].")
        return gt_projected_vertices, gt_vertex_rel_depths

    def _normalize_gt_vertices(
        self,
        gt_vertices: torch.Tensor,
        num_frames: int,
        num_vertices: int,
    ) -> torch.Tensor:
        if gt_vertices.ndim == 2:
            gt_vertices = gt_vertices.unsqueeze(0)
        elif gt_vertices.ndim != 3:
            raise ValueError("gt_vertices must be [V,3] or [T,V,3].")
        if gt_vertices.shape[0] != num_frames:
            raise ValueError("gt_vertices must match the number of frames.")
        if gt_vertices.shape[1] != num_vertices or gt_vertices.shape[2] != 3:
            raise ValueError("gt_vertices must be [T,V,3] and match batched_vertices.")
        return gt_vertices

    def _normalize_gt_log_vars(
        self,
        gt_log_vars: torch.Tensor,
        num_frames: int,
        num_views: int,
        num_vertices: int,
        name: str,
    ) -> torch.Tensor:
        if gt_log_vars.ndim == 1:
            gt_log_vars = gt_log_vars.view(1, 1, -1, 1)
        elif gt_log_vars.ndim == 2:
            if gt_log_vars.shape[1] == 1:
                gt_log_vars = gt_log_vars.unsqueeze(0).unsqueeze(0)
            elif gt_log_vars.shape[0] == num_frames and gt_log_vars.shape[1] == num_vertices:
                gt_log_vars = gt_log_vars.unsqueeze(1).unsqueeze(-1)
            elif num_frames == 1 and gt_log_vars.shape[0] == num_views and gt_log_vars.shape[1] == num_vertices:
                gt_log_vars = gt_log_vars.unsqueeze(0).unsqueeze(-1)
            else:
                raise ValueError(f"Unsupported {name} shape.")
        elif gt_log_vars.ndim == 3:
            if gt_log_vars.shape[-1] == 1:
                if gt_log_vars.shape[0] == num_frames and gt_log_vars.shape[1] == num_vertices:
                    gt_log_vars = gt_log_vars.unsqueeze(1)
                elif num_frames == 1 and gt_log_vars.shape[0] == num_views and gt_log_vars.shape[1] == num_vertices:
                    gt_log_vars = gt_log_vars.unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported {name} shape.")
            else:
                if (
                    gt_log_vars.shape[0] == num_frames
                    and gt_log_vars.shape[1] == num_views
                    and gt_log_vars.shape[2] == num_vertices
                ):
                    gt_log_vars = gt_log_vars.unsqueeze(-1)
                else:
                    raise ValueError(f"Unsupported {name} shape.")
        elif gt_log_vars.ndim != 4:
            raise ValueError(
                f"{name} must be [V], [V,1], [T,V], [T,V,1], [C,V], [C,V,1], [T,C,V], or [T,C,V,1]."
            )
        if gt_log_vars.shape[-1] != 1:
            raise ValueError(f"{name} must end with size 1.")
        if tuple(gt_log_vars.shape[:3]) != (num_frames, num_views, num_vertices):
            raise ValueError(f"{name} must match [T,C,V].")
        return gt_log_vars

    def assemble_x_system(
        self,
        view_mat: torch.Tensor,
        viewproj_mat: torch.Tensor,
        batched_vertices: torch.Tensor,
        batched_joints: torch.Tensor,
        gt_projected_vertices: torch.Tensor,
        gt_vertex_rel_depths: torch.Tensor,
        gt_projected_log_vars: torch.Tensor | None,
        gt_vertex_rel_depth_log_vars: torch.Tensor | None,
        batched_x: torch.Tensor,
        batched_v_jacobian: torch.Tensor,
        batched_j_jacobian: torch.Tensor,
        alignment_weight: float,
        rel_depth_weight: float,
        exp_reg_weight: float,
        pose_reg_weight: float,
        vertex_indices: torch.Tensor,
    ):
        squeeze_frame = False
        if batched_x.ndim == 1:
            squeeze_frame = True
            batched_x = batched_x.unsqueeze(0)
        elif batched_x.ndim != 2:
            raise ValueError(f"batched_x must be [{self.dim_x}] or [T,{self.dim_x}].")

        if batched_vertices.ndim == 2:
            batched_vertices = batched_vertices.unsqueeze(0)
        if batched_joints.ndim == 2:
            batched_joints = batched_joints.unsqueeze(0)
        if batched_v_jacobian.ndim == 3:
            batched_v_jacobian = batched_v_jacobian.unsqueeze(0)
        if batched_j_jacobian.ndim == 3:
            batched_j_jacobian = batched_j_jacobian.unsqueeze(0)

        num_frames = batched_x.shape[0]
        if (
            batched_vertices.ndim != 3
            or batched_joints.ndim != 3
            or batched_v_jacobian.ndim != 4
            or batched_j_jacobian.ndim != 4
            or batched_vertices.shape[0] != num_frames
            or batched_joints.shape[0] != num_frames
            or batched_v_jacobian.shape[0] != num_frames
            or batched_j_jacobian.shape[0] != num_frames
        ):
            raise ValueError("FLAME tensors must match [T,...] layout.")

        view_mat, viewproj_mat = self._normalize_camera(view_mat, viewproj_mat)
        num_views = int(view_mat.shape[-3])
        gt_projected_vertices, gt_vertex_rel_depths = self._normalize_gt_observations(
            gt_projected_vertices,
            gt_vertex_rel_depths,
            num_frames,
            num_views,
        )
        num_vertices = int(gt_projected_vertices.shape[2])
        if gt_projected_log_vars is None and gt_vertex_rel_depth_log_vars is None:
            empty = torch.empty((0,), dtype=torch.float32, device=gt_projected_vertices.device)
            gt_projected_log_vars = empty
            gt_vertex_rel_depth_log_vars = empty
        elif gt_projected_log_vars is None or gt_vertex_rel_depth_log_vars is None:
            raise ValueError("gt_projected_log_vars and gt_vertex_rel_depth_log_vars must both be provided or both be None.")
        else:
            gt_projected_log_vars = self._normalize_gt_log_vars(
                gt_projected_log_vars,
                num_frames,
                num_views,
                num_vertices,
                "gt_projected_log_vars",
            )
            gt_vertex_rel_depth_log_vars = self._normalize_gt_log_vars(
                gt_vertex_rel_depth_log_vars,
                num_frames,
                num_views,
                num_vertices,
                "gt_vertex_rel_depth_log_vars",
            )

        residual, jacobian = self.extension.assemble_x_jacobian(
            view_mat,
            viewproj_mat,
            batched_vertices,
            batched_joints,
            gt_projected_vertices,
            gt_vertex_rel_depths,
            gt_projected_log_vars,
            gt_vertex_rel_depth_log_vars,
            batched_x,
            batched_v_jacobian,
            batched_j_jacobian,
            alignment_weight,
            rel_depth_weight,
            exp_reg_weight,
            pose_reg_weight,
            vertex_indices,
            self.num_expressions,
        )
        if squeeze_frame:
            return residual[0], jacobian[0]
        return residual, jacobian

    def assemble_identity_system(
        self,
        view_mat: torch.Tensor,
        viewproj_mat: torch.Tensor,
        batched_vertices: torch.Tensor,
        batched_joints: torch.Tensor,
        gt_projected_vertices: torch.Tensor,
        gt_vertex_rel_depths: torch.Tensor,
        gt_projected_log_vars: torch.Tensor | None,
        gt_vertex_rel_depth_log_vars: torch.Tensor | None,
        identity: torch.Tensor,
        batched_v_jacobian: torch.Tensor,
        batched_j_jacobian: torch.Tensor,
        alignment_weight: float,
        rel_depth_weight: float,
        identity_reg_weight: float,
        vertex_indices: torch.Tensor,
    ):
        if batched_vertices.ndim == 2:
            batched_vertices = batched_vertices.unsqueeze(0)
        if batched_joints.ndim == 2:
            batched_joints = batched_joints.unsqueeze(0)
        if batched_v_jacobian.ndim == 3:
            batched_v_jacobian = batched_v_jacobian.unsqueeze(0)
        if batched_j_jacobian.ndim == 3:
            batched_j_jacobian = batched_j_jacobian.unsqueeze(0)

        if (
            batched_vertices.ndim != 3
            or batched_joints.ndim != 3
            or batched_v_jacobian.ndim != 4
            or batched_j_jacobian.ndim != 4
        ):
            raise ValueError("FLAME tensors must be [T,...] or single-frame equivalents.")
        num_frames = int(batched_vertices.shape[0])
        if (
            batched_joints.shape[0] != num_frames
            or batched_v_jacobian.shape[0] != num_frames
            or batched_j_jacobian.shape[0] != num_frames
        ):
            raise ValueError("FLAME tensors must share the same frame count.")

        view_mat, viewproj_mat = self._normalize_camera(view_mat, viewproj_mat)
        num_views = int(view_mat.shape[-3])
        gt_projected_vertices, gt_vertex_rel_depths = self._normalize_gt_observations(
            gt_projected_vertices,
            gt_vertex_rel_depths,
            num_frames,
            num_views,
        )
        num_vertices = int(gt_projected_vertices.shape[2])
        if gt_projected_log_vars is None and gt_vertex_rel_depth_log_vars is None:
            empty = torch.empty((0,), dtype=torch.float32, device=gt_projected_vertices.device)
            gt_projected_log_vars = empty
            gt_vertex_rel_depth_log_vars = empty
        elif gt_projected_log_vars is None or gt_vertex_rel_depth_log_vars is None:
            raise ValueError("gt_projected_log_vars and gt_vertex_rel_depth_log_vars must both be provided or both be None.")
        else:
            gt_projected_log_vars = self._normalize_gt_log_vars(
                gt_projected_log_vars,
                num_frames,
                num_views,
                num_vertices,
                "gt_projected_log_vars",
            )
            gt_vertex_rel_depth_log_vars = self._normalize_gt_log_vars(
                gt_vertex_rel_depth_log_vars,
                num_frames,
                num_views,
                num_vertices,
                "gt_vertex_rel_depth_log_vars",
            )
        identity = identity.reshape(-1)
        return self.extension.assemble_identity_jacobian(
            view_mat,
            viewproj_mat,
            batched_vertices,
            batched_joints,
            gt_projected_vertices,
            gt_vertex_rel_depths,
            gt_projected_log_vars,
            gt_vertex_rel_depth_log_vars,
            identity,
            batched_v_jacobian,
            batched_j_jacobian,
            alignment_weight,
            rel_depth_weight,
            identity_reg_weight,
            vertex_indices,
        )

    def assemble_x_direct_system(
        self,
        gt_vertices: torch.Tensor,
        batched_vertices: torch.Tensor,
        batched_x: torch.Tensor,
        batched_v_jacobian: torch.Tensor,
        vertex_weight: float,
        exp_reg_weight: float,
        pose_reg_weight: float,
        vertex_indices: torch.Tensor,
    ):
        squeeze_frame = False
        if batched_x.ndim == 1:
            squeeze_frame = True
            batched_x = batched_x.unsqueeze(0)
        elif batched_x.ndim != 2:
            raise ValueError(f"batched_x must be [{self.dim_x}] or [T,{self.dim_x}].")

        if batched_vertices.ndim == 2:
            batched_vertices = batched_vertices.unsqueeze(0)
        if batched_v_jacobian.ndim == 3:
            batched_v_jacobian = batched_v_jacobian.unsqueeze(0)

        num_frames = batched_x.shape[0]
        if (
            batched_vertices.ndim != 3
            or batched_v_jacobian.ndim != 4
            or batched_vertices.shape[0] != num_frames
            or batched_v_jacobian.shape[0] != num_frames
        ):
            raise ValueError("FLAME tensors must match [T,...] layout.")

        num_vertices = int(batched_vertices.shape[1])
        gt_vertices = self._normalize_gt_vertices(gt_vertices, num_frames, num_vertices)
        residual, jacobian = self.extension.assemble_x_direct_jacobian(
            gt_vertices,
            batched_x,
            batched_vertices,
            batched_v_jacobian,
            vertex_weight,
            exp_reg_weight,
            pose_reg_weight,
            vertex_indices,
            self.num_expressions,
        )
        if squeeze_frame:
            return residual[0], jacobian[0]
        return residual, jacobian

    def assemble_identity_direct_system(
        self,
        gt_vertices: torch.Tensor,
        batched_vertices: torch.Tensor,
        identity: torch.Tensor,
        batched_v_jacobian: torch.Tensor,
        vertex_weight: float,
        identity_reg_weight: float,
        vertex_indices: torch.Tensor,
    ):
        if batched_vertices.ndim == 2:
            batched_vertices = batched_vertices.unsqueeze(0)
        if batched_v_jacobian.ndim == 3:
            batched_v_jacobian = batched_v_jacobian.unsqueeze(0)

        if batched_vertices.ndim != 3 or batched_v_jacobian.ndim != 4:
            raise ValueError("FLAME tensors must be [T,...] or single-frame equivalents.")
        num_frames = int(batched_vertices.shape[0])
        if batched_v_jacobian.shape[0] != num_frames:
            raise ValueError("FLAME tensors must share the same frame count.")

        num_vertices = int(batched_vertices.shape[1])
        gt_vertices = self._normalize_gt_vertices(gt_vertices, num_frames, num_vertices)
        identity = identity.reshape(-1)
        return self.extension.assemble_identity_direct_jacobian(
            gt_vertices,
            identity,
            batched_vertices,
            batched_v_jacobian,
            vertex_weight,
            identity_reg_weight,
            vertex_indices,
        )

    def solve_normal_equation(self, context, jacobian, residual, damping_lambda: float):
        return self.extension.cholesky_solve(
            context.cublas_handle,
            context.cusolver_handle,
            jacobian,
            residual,
            damping_lambda,
        )
