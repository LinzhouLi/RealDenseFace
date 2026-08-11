from itertools import product
from math import ceil

import numpy as np
import torch

from .model import FaceBoxes


class PriorBox:
    def __init__(self, image_size):
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0, j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0, i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        anchors += [
                            (j + 0.5) * self.steps[k] / self.image_size[1],
                            (i + 0.5) * self.steps[k] / self.image_size[0],
                            s_kx,
                            s_ky,
                        ]

        output = torch.tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class FaceDetector:
    def __init__(
        self,
        model_path: str,
        image_size: tuple[int, int],
        threshold: float = 0.6,
        half: bool = False,
        device: str = "cuda",
        compile_model: bool = True,
    ):
        self.threshold = threshold
        self.image_size = image_size
        self.variance = [0.1, 0.2]
        self.dtype = torch.float16 if half else torch.float32
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = FaceBoxes(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        if half:
            self.model.half()

        self.box_scale = np.array([image_size[1], image_size[0], image_size[1], image_size[0]], dtype=np.float32)
        self.image_mean = torch.tensor([104, 117, 123], dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        self.priors = PriorBox(image_size).forward().to(dtype=self.dtype, device=self.device)
        self.priors_cpu = self.priors.float().cpu().numpy()
        if compile_model and hasattr(torch, "compile"):
            self._compile()

    def _preprocess(self, image: torch.Tensor):
        image = image.flip(1)
        image = image * 255.0 - self.image_mean
        loc, conf = self.model(image)
        return loc[0], conf[0, :, 1]

    @torch.no_grad()
    def _compile(self):
        self._preprocess = torch.compile(self._preprocess, mode="max-autotune", dynamic=False)
        dummy = torch.rand([1, 3, self.image_size[0], self.image_size[1]], dtype=self.dtype, device=self.device)
        for _ in range(3):
            self._preprocess(dummy)

    @torch.no_grad()
    def detect_one(self, image: torch.Tensor):
        _, _, height, width = image.shape
        if (height, width) != self.image_size:
            raise ValueError(f"Expected input size {self.image_size}, got {(height, width)}")

        loc, conf = self._preprocess(image.to(device=self.device, dtype=self.dtype))
        best_idx = int(torch.argmax(conf))
        score = float(conf[best_idx].item())
        loc = loc[best_idx].float().cpu().numpy()
        box = self.priors_cpu[best_idx].copy()
        box[:2] = box[:2] + loc[:2] * self.variance[0] * box[2:]
        box[2:] = box[2:] * np.exp(loc[2:] * self.variance[1])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        return (box * self.box_scale + 0.5).astype(np.int32).tolist(), score
