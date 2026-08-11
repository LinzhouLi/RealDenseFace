import pickle
from typing import Optional, Union

import numpy as np
import torch
from roma import rotvec_to_rotmat

from common.config import FLAMEConfig
from common.utils import load_expression_blendshapes


def _matmul_acc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-3)
    return (a * b).sum(-2)


def _affine_transform(transform: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    transform_rot = transform[..., :3, :3]
    transform_trans = transform[..., :3, 3]
    return (transform_rot * position.unsqueeze(-2)).sum(-1) + transform_trans


class FLAME:
    def __init__(
        self,
        config: Union[FLAMEConfig, str],
        simplified: bool = False,
        device: str | torch.device | None = None,
    ):
        if isinstance(config, str):
            config = FLAMEConfig(flame_model_path=config)
        elif not isinstance(config, FLAMEConfig):
            raise TypeError(f"FLAME expects FLAMEConfig or flame_model_path string, got {type(config).__name__}.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        with open(config.flame_model_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self.simplified = simplified
        self.device = device
        self.dtype = torch.float32
        self.config = config
        self.num_expressions = config.resolved_num_expressions()
        self.v_template = torch.from_numpy(np.asarray(data["v_template"])).to(device=device, dtype=self.dtype)
        self.faces = torch.from_numpy(np.asarray(data["f"]).astype(np.int64)).to(device=device, dtype=torch.int64)
        self.pose_dirs = torch.from_numpy(np.asarray(data["posedirs"])).to(device=device, dtype=self.dtype)
        self.lbs_weights = torch.from_numpy(np.asarray(data["weights"])).to(device=device, dtype=self.dtype)
        self.parents = torch.from_numpy(np.asarray(data["kintree_table"][0])).to(device=device, dtype=torch.int64)
        self.J_regressor = torch.from_numpy(np.asarray(data["J_regressor"].todense())).to(device=device, dtype=self.dtype)

        shape_dirs_np = np.asarray(data["shapedirs"])[..., :300].astype(np.float32, copy=False)
        self.shape_dirs = torch.from_numpy(np.ascontiguousarray(shape_dirs_np)).to(
            device=device, dtype=self.dtype
        ).view(-1, 300).T.contiguous()
        expr_dirs_np = load_expression_blendshapes(data, config)
        self.expr_dirs = torch.from_numpy(expr_dirs_np).to(device=device, dtype=self.dtype) \
            .view(-1, self.num_expressions).T.contiguous()
        num_pose_basis = self.pose_dirs.shape[-1]
        self.pose_dirs = self.pose_dirs.reshape([-1, num_pose_basis]).T.contiguous()
        self.joints_dirs = (
            self.J_regressor
            @ self.shape_dirs.view(300, self.v_template.shape[0], 3).permute(1, 2, 0).reshape(self.v_template.shape[0], 300 * 3)
        ).view(5, 3, 300).permute(2, 0, 1).contiguous()

    def calc_canonical(self, identity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if identity.ndim == 1:
            v_canonical = self.v_template + torch.matmul(identity, self.shape_dirs).view(-1, 3)
            j_canonical = torch.matmul(self.J_regressor, v_canonical)
            return v_canonical, j_canonical

        v_canonical = self.v_template.unsqueeze(0) + torch.matmul(identity, self.shape_dirs).view(identity.shape[0], -1, 3)
        j_canonical = torch.matmul(self.J_regressor.unsqueeze(0).expand(identity.shape[0], -1, -1), v_canonical)
        return v_canonical, j_canonical

    def forward_from_canonical(
        self,
        v_canonical: torch.Tensor,
        j_canonical: torch.Tensor,
        x: torch.Tensor,
        return_joints: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        add_batch = False
        if x.ndim == 1:
            add_batch = True
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        if v_canonical.ndim == 2:
            v_canonical = v_canonical.unsqueeze(0).expand(batch_size, -1, -1)
        if j_canonical.ndim == 2:
            j_canonical = j_canonical.unsqueeze(0).expand(batch_size, -1, -1)

        n_expr = self.num_expressions
        expression = x[:, :n_expr].reshape(batch_size, n_expr)
        pose = x[:, n_expr:n_expr + 15].reshape(batch_size, 5, 3)
        translation = x[:, n_expr + 15:n_expr + 18].reshape(batch_size, 3)

        v_shaped = v_canonical + torch.matmul(expression, self.expr_dirs).view(batch_size, -1, 3)
        if self.simplified:
            joints = j_canonical
        else:
            joints = torch.matmul(self.J_regressor.unsqueeze(0).expand(batch_size, -1, -1), v_shaped)

        rot_mats = rotvec_to_rotmat(pose)
        if self.simplified:
            v_posed = v_shaped
        else:
            identity_mat = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(0)
            pose_feat = (rot_mats[:, 1:] - identity_mat).reshape(batch_size, -1)
            v_posed = v_shaped + torch.matmul(pose_feat, self.pose_dirs).view(batch_size, -1, 3)

        rel_joints = joints.clone()
        rel_joints[:, 1] -= joints[:, 0]
        rel_joints[:, 2:] -= joints[:, 1:2]

        rel_affine_mats = torch.zeros(batch_size, 5, 4, 4, device=self.device, dtype=self.dtype)
        rel_affine_mats[..., :3, :3] = rot_mats
        rel_affine_mats[..., :3, 3] = rel_joints
        rel_affine_mats[..., 3, 3] = 1.0

        affine_mats = rel_affine_mats.clone()
        affine_mats[:, 1] = _matmul_acc(affine_mats[:, 0].clone(), rel_affine_mats[:, 1])
        affine_mats[:, 2:] = _matmul_acc(affine_mats[:, 1:2].clone(), rel_affine_mats[:, 2:])

        result_joints = affine_mats[..., :3, 3].clone()
        joint_transformed = (affine_mats[..., :3, :3] * joints.unsqueeze(-2)).sum(-1)
        affine_mats = affine_mats.clone()
        affine_mats[..., :3, 3] -= joint_transformed

        transforms = _matmul_acc(self.lbs_weights.unsqueeze(0), affine_mats.view(batch_size, 5, 16)).view(batch_size, -1, 4, 4)
        v_skinned = _affine_transform(transforms, v_posed)
        v_skinned += translation.unsqueeze(1)
        result_joints += translation.unsqueeze(1)

        if add_batch:
            v_skinned = v_skinned.squeeze(0)
            result_joints = result_joints.squeeze(0)
        if return_joints:
            return v_skinned, result_joints
        return v_skinned

    def forward(
        self,
        identity: torch.Tensor,
        x: torch.Tensor,
        v_offsets: Optional[torch.Tensor] = None,
        return_joints: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        v_canonical, j_canonical = self.calc_canonical(identity)
        vertices, joints = self.forward_from_canonical(v_canonical, j_canonical, x, return_joints=True)
        if v_offsets is not None:
            if vertices.ndim == 2 and v_offsets.ndim == 2:
                vertices = vertices + v_offsets
            elif vertices.ndim == 3 and v_offsets.ndim == 2:
                vertices = vertices + v_offsets.unsqueeze(0)
            else:
                vertices = vertices + v_offsets
        if return_joints:
            return vertices, joints
        return vertices
