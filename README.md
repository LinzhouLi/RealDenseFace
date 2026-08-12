<h1 align="center">RealDenseFace</h1>

<p align="center">
  <strong>Real-time Monocular 3D Face Reconstruction from Dense UV-space Priors</strong>
</p>

<p align="center">
  Linzhou Li &nbsp;&nbsp; Tianjia Shao &nbsp;&nbsp; Kun Zhou<br>
  State Key Lab of CAD&amp;CG, Zhejiang University
</p>

<p align="center">
  <a href="https://gapszju.github.io/RealDenseFace/">Project Page</a>
  &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2608.09238">Paper</a>
</p>

<p align="center">
  <img src="assets/teaser.jpg" width="100%" alt="RealDenseFace teaser">
</p>

RealDenseFace is a real-time optimization-based framework for monocular FLAME
reconstruction. It predicts dense UV-space correspondence and relative-depth
priors, and fits FLAME using a tailored CUDA Gauss-Newton solver.

This repository provides inference and fitting code for single images, offline
monocular sequences, online tracking, and a NeRSemble v2 multi-view tracking
example.

## News

- **2026-08-12**: Inference and fitting code with pretrained models released.
- **2026-08-11**: Paper and project page released.

Coming soon:

- GUI demo
- Training data

## Installation

### Requirements

- Python 3.11
- PyTorch with CUDA support
- An NVIDIA GPU and a CUDA toolkit compatible with the installed PyTorch build

Create a conda environment:

```bash
conda create -n realdenseface python=3.11
conda activate realdenseface
```

Install a CUDA-enabled PyTorch build by following the
[official PyTorch instructions](https://pytorch.org/get-started/locally/). Then
install the remaining dependencies and build the local CUDA solver:

```bash
pip install -r requirements.txt
pip install -e ./cuda_extensions/flame_solver --no-build-isolation
```

### Pretrained models

Download the released ViT-S and ViT-B checkpoints from
[Google Drive](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view?usp=sharing) and place them under `weights/`:

### FLAME model

The official FLAME model is subject to its own license and is not redistributed
in this repository. Download `flame2023.pkl` from the
[official FLAME website](https://flame.is.tue.mpg.de/) and place it at:

```text
weights/flame/flame2023.pkl
```

The expected model and asset layout is:

```text
weights/
├── facebox/
│   └── face_box.pth
├── flame/
│   ├── flame2023.pkl
│   └── flame_assets.npz
└── realdenseface/
    ├── vitb.pth
    └── vits.pth
```

Run all commands below from the repository root.

## Usage

### Single-image fitting

```bash
python fit_single_image.py \
    --input path/to/image.jpg \
    --output_npz output/single_image.npz \
    --output_image output/single_image.jpg
```

Single-image fitting uses the ViT-B checkpoint by default. To use ViT-S:

```bash
python fit_single_image.py \
    --input path/to/image.jpg \
    --output_npz output/single_image.npz \
    --output_image output/single_image.jpg \
    --model_config configs/model/vits.yaml \
    --model_weights weights/realdenseface/vits.pth
```

### Offline monocular video fitting

Offline fitting processes the complete sequence and refines a shared identity
from multiple frames:

```bash
python track_video_offline.py \
    --input assets/demo_video.mp4 \
    --output_npz output/offline_tracking.npz \
    --output_mp4 output/offline_visualization.mp4
```

### Online monocular video tracking

Online tracking processes frames sequentially without using future frames. It
uses the ViT-S checkpoint by default:

```bash
python track_video_online.py \
    --input assets/demo_video.mp4 \
    --output_mp4 output/online_visualization.mp4
```

### NeRSemble v2 multi-view tracking

The NeRSemble v2 example downsamples the 16 camera videos, detects temporal
jumps, caches RealDenseFace predictions, fits a multi-view FLAME sequence, and
renders tracking visualizations:

```bash
python preprocess_nersemble_v2.py \
    --dataset_root path/to/nersemble_v2 \
    --subjects 001 \
    --stride 1
```

For each sequence, results are written inside the dataset:

```text
<dataset_root>/<subject>/flame_tracking/<sequence>/
├── downsample_videos/
├── inference_cache/
├── jump_frames.npz
├── tracking_results.npz
└── visualization/
```

Model architecture configurations are stored under `configs/model/`, while
task-specific fitting parameters are stored separately under `configs/fitting/`.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026realdenseface,
  title   = {RealDenseFace: Real-time Monocular 3D Face Reconstruction from Dense UV-space Priors},
  author  = {Li, Linzhou and Shao, Tianjia and Zhou, Kun},
  journal = {arXiv preprint arXiv:2608.09238},
  year    = {2026}
}
```
