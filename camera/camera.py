import math
import torch
from torch import nn
import numpy as np


def fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal: float, pixels: float) -> float:
    return 2 * math.atan(pixels / (2 * focal))


class Camera(nn.Module):
    def __init__(
        self, 
        rot: np.ndarray = None,   # rotation matrix, shape: [3, 3]
        pos: np.ndarray = None,   # translation vector, shape: [3]
        width: int = 1024, 
        height: int = 1024, 
        znear: float = 0.001, 
        zfar: float = 100.0
    ) -> None:
        super(Camera, self).__init__()

        self.znear = znear
        self.zfar = zfar
        self.width = width
        self.height = height

        self._cx = None
        self._cy = None
        self.fov_x = None
        self.fov_y = None

        if rot is None:
            rot = np.eye(3)
        if pos is None:
            pos = np.zeros([3])

        rot = rot.astype(np.float32) # caution the type from numpy
        pos = pos.astype(np.float32)
        
        self.register_buffer('_proj', torch.eye(4, dtype=torch.float32))
        self.register_buffer('_w2v', torch.eye(4, dtype=torch.float32))
        self.register_buffer('_v2w', torch.eye(4, dtype=torch.float32))
        self.register_buffer('_rot', torch.from_numpy(rot))
        self.register_buffer('_pos', torch.from_numpy(pos))

        self._update_world2view_matrix()
        self._update_view2world_matrix()

    @property
    def cx(self):
        return (self.width - 1) / 2 if self._cx is None else self._cx # caution: default width
    
    @property
    def cy(self):
        return (self.height - 1) / 2 if self._cy is None else self._cy # caution: default height
    
    @property
    def fx(self):
        return fov2focal(self.fov_x, self.width)
    
    @property
    def fy(self):
        return fov2focal(self.fov_y, self.height)
    
    @property
    def intrisics(self):
        return torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=self._proj.dtype, device=self._proj.device)
    
    @property
    def extrinsics(self):
        return self._w2v
    
    @property
    def get_pos(self) -> torch.Tensor:
        return self._pos
    
    @property
    def get_w2v(self) -> torch.Tensor:
        return self._w2v
    
    @property
    def get_v2w(self) -> torch.Tensor:
        return self._v2w
    
    @property
    def get_proj(self) -> torch.Tensor:
        return self._proj
    
    @property
    def get_full_proj(self) -> torch.Tensor:
        return torch.matmul(self._proj, self._w2v)
    
    def _update_projection_matrix(self) -> torch.Tensor:
        n, f = self.znear, self.zfar
        w, h = self.width, self.height

        P = torch.zeros_like(self._proj)
        P[0, 0] = 2 * self.fx / w
        P[1, 1] = 2 * self.fy / h

        # consider camera center cx, cy
        P[0, 2] = -1 + 2 * (self.cx / w)
        P[1, 2] = -1 + 2 * (self.cy / h)

        # z = zfar, depth = 1.0; z = znear, depth = 0.0
        P[3, 2] = 1.0
        P[2, 2] = f / (f - n)
        P[2, 3] = -(f * n) / (f - n)
        self._proj.copy_(P)

    def _update_view2world_matrix(self) -> torch.Tensor:
        self._v2w[:3, :3].copy_(self._rot)
        self._v2w[:3, 3].copy_(self._pos)

    def _update_world2view_matrix(self) -> torch.Tensor:
        R = self._rot.transpose(0, 1)
        T = -torch.matmul(R, self._pos)
        self._w2v[:3, :3].copy_(R)
        self._w2v[:3, 3].copy_(T)
    
    def set_pose(self, pose: torch.Tensor | np.ndarray) -> None:
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).to(self._rot.device, self._rot.dtype)
        self._rot.copy_(pose[:3, :3])
        self._pos.copy_(pose[:3, 3])
        self._update_view2world_matrix()
        self._update_world2view_matrix()
    
    def project_points(self, points: torch.Tensor, in_viewport: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        points_pad = torch.nn.functional.pad(points, [0, 1], value=1.0)
        proj_mat = self.get_full_proj.to(points_pad.device)
        points_proj = torch.matmul(points_pad, proj_mat.transpose(0, 1))
        points_proj = points_proj[..., :3] / torch.clamp_min(points_proj[..., 3:4], 1e-8)
        filter_mask = torch.logical_and(points_proj[..., 2] > 0.0, points_proj[..., 2] < 1.0)
        if in_viewport:
            points_proj[..., :2] = points_proj[..., :2] * 0.5 + 0.5
            points_proj[..., 0] = points_proj[..., 0] * self.width
            points_proj[..., 1] = points_proj[..., 1] * self.height
        return points_proj, filter_mask
    
    def clone(self):
        new_camera = Camera(
            rot=self._rot.cpu().numpy(), 
            pos=self._pos.cpu().numpy(),
            width=self.width, height=self.height,
            znear=self.znear, zfar=self.zfar
        )
        new_camera._cx = self._cx
        new_camera._cy = self._cy
        new_camera.fov_x = self.fov_x
        new_camera.fov_y = self.fov_y
        new_camera._proj.copy_(self._proj)
        new_camera._w2v.copy_(self._w2v)
        new_camera._v2w.copy_(self._v2w)
        return new_camera

    def crop(self, x_min: int, y_min: int, x_max: int, y_max: int):
        # assert x_min >= 0 and x_max <= self.width # NOTE: crop may go beyond image border
        # assert y_min >= 0 and y_max <= self.height

        new_width = x_max - x_min
        new_height = y_max - y_min
        new_fov_x = focal2fov(self.fx, new_width)
        new_fov_y = focal2fov(self.fy, new_height)
        adjust_position = False
        if adjust_position: # TODO: some bugs here
            new_cx = self.cx / self.width * new_width
            new_cy = self.cy / self.height * new_height
            u_c_new = new_cx + x_min
            v_c_new = new_cy + y_min
            shift_cam = torch.tensor([(u_c_new - self.cx) / self.fx, (v_c_new - self.cy) / self.fy, 0.0], device=self._pos.device, dtype=self._pos.dtype)
            shift_world = torch.matmul(self._rot, shift_cam)
            self._pos += shift_world
            self._update_world2view_matrix()
            self._update_view2world_matrix()
        else:
            new_cx = self.cx - x_min
            new_cy = self.cy - y_min

        self._cx = new_cx
        self._cy = new_cy
        self.fov_x = new_fov_x
        self.fov_y = new_fov_y
        self.width = new_width
        self.height = new_height
        self._update_projection_matrix()

    def resize(self, width: int, height: int):
        if self._cx is not None: self._cx = self._cx / self.width * width
        if self._cy is not None: self._cy = self._cy / self.height * height
        self.width = width
        self.height = height
        self._update_projection_matrix()

    def flip(self, axis: str = 'x'): # Caution: change winding order
        if axis == 'x':
            self._rot[:, 0] *= -1
            self._cx = self.width - self.cx
        elif axis == 'y':
            self._rot[:, 1] *= -1
            self._cy = self.height - self.cy
        else:
            raise ValueError("axis must be 'x' or 'y'")

        self._update_world2view_matrix()
        self._update_view2world_matrix()
        self._update_projection_matrix()

    def rotate90(self, clockwise: bool = True):
        device = self._rot.device
        dtype = self._rot.dtype
        if clockwise:
            trans_mat = torch.tensor([
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
            ], device=device, dtype=dtype)
        else:
            trans_mat = torch.tensor([
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
            ], device=device, dtype=dtype)

        new_rot = torch.matmul(self._rot, trans_mat)
        self._rot.copy_(new_rot)

        old_w, old_h = self.width, self.height
        old_cx, old_cy = self.cx, self.cy
        old_fov_x, old_fov_y = self.fov_x, self.fov_y

        self.width, self.height = old_h, old_w
        self.fov_x, self.fov_y = old_fov_y, old_fov_x

        if clockwise: # (x, y) -> (y, W-x)
            self._cx = old_h - old_cy
            self._cy = old_cx
        else: # (x, y) -> (H-y, x)
            self._cx = old_cy
            self._cy = old_w - old_cx

        self._update_world2view_matrix()
        self._update_view2world_matrix()
        self._update_projection_matrix()
    
    def print(self):
        print("Camera Parameters:")
        print("width:", self.width, "height:", self.height)
        print("focal x:", self.fx, "focal y:", self.fy)
        print("fov x:", self.fov_x, "fov y:", self.fov_y)
        print("principal point x:", self.cx, "principal point y:", self.cy)
        print("position:", self.get_pos)
        print("projection matrix:", self.get_proj)
        print("view2world matrix:", self.get_v2w)
        print("world2view matrix:", self.get_w2v)


class PerspectiveCamera(Camera):
    def __init__(
        self,
        fov_y: float = np.pi / 3, 
        rot: np.ndarray = None,   # rotation matrix, shape: [3, 3]
        pos: np.ndarray = None,   # translation vector, shape: [3]
        width: int = 1024, 
        height: int = 1024, 
        znear: float = 0.01, 
        zfar: float = 100.0
    ):
        super(PerspectiveCamera, self).__init__(
            rot=rot, pos=pos,
            width=width, height=height,
            znear=znear, zfar=zfar
        )

        self.fov_y = fov_y
        focal = fov2focal(fov_y, height)
        self.fov_x = focal2fov(focal, width) # caution: compute fovx from fovy
        self._update_projection_matrix()

    def set_fov_y(self, fov_y: float) -> None:
        self.fov_y = float(fov_y)
        focal = fov2focal(self.fov_y, self.height)
        self.fov_x = focal2fov(focal, self.width) # caution: compute fovx from fovy
        self._update_projection_matrix()

    def resize(self, width: int, height: int):
        super().resize(width, height)
        # keep vertical fov fixed, recompute horizontal fov for the new aspect
        focal = fov2focal(self.fov_y, self.height)
        self.fov_x = focal2fov(focal, self.width)
        self._update_projection_matrix()


class IntrinsicsCamera(Camera):
    def __init__(
        self, 
        K: np.ndarray = None,   # intrinsic parameters, shape: [3, 3]
        R: np.ndarray = None,   # extrinsic rotation matrix, shape: [3, 3]
        T: np.ndarray = None,   # extrinsic translation vector, shape: [3]
        D: np.ndarray = None,   # distortion
        width: int = 1024, 
        height: int = 1024, 
        znear: float = 0.01,
        zfar: float = 100
    ):
        # world2view -> view2world
        rot = np.transpose(R)
        pos = -np.dot(rot, T)

        super(IntrinsicsCamera, self).__init__(
            rot=rot,
            pos=pos,
            width=width, height=height,
            znear=znear, zfar=zfar
        )

        fx, fy, self._cx, self._cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item() # add item() to convert numpy scalar to float (some numpy version issue)
        self.fov_x, self.fov_y = focal2fov(fx, width), focal2fov(fy, height)
        self.K = K
        self.D = D
        self._update_projection_matrix()


class OrthographicCamera(Camera):
    def __init__(
        self,
        xmag: float = 1.0,
        ymag: float | None = None,
        rot: np.ndarray = None,
        pos: np.ndarray = None,
        width: int = 1024,
        height: int = 1024,
        cx: float | None = None,
        cy: float | None = None,
        znear: float = -100.0,
        zfar: float = 100.0,
    ):
        super(OrthographicCamera, self).__init__(
            rot=rot,
            pos=pos,
            width=width,
            height=height,
            znear=znear,
            zfar=zfar,
        )

        self.xmag = float(xmag)
        self.ymag = float(xmag if ymag is None else ymag)
        self._cx = (width - 1) * 0.5 if cx is None else float(cx)
        self._cy = (height - 1) * 0.5 if cy is None else float(cy)
        self._update_projection_matrix()

    def _update_projection_matrix(self) -> torch.Tensor:
        n, f = self.znear, self.zfar
        w, h = self.width, self.height

        P = torch.zeros_like(self._proj)
        P[0, 0] = 1.0 / self.xmag
        P[1, 1] = 1.0 / self.ymag
        P[0, 3] = (w - 2.0 * self.cx) / w
        P[1, 3] = (h - 2.0 * self.cy) / h
        P[2, 2] = 1.0 / (f - n)
        P[2, 3] = -n / (f - n)
        P[3, 3] = 1.0
        self._proj.copy_(P)
