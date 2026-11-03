---
title: RealDenseFace
emoji: 🎭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.31.0"
python_version: "3.12"
app_file: app.py
startup_duration_timeout: 1h
license: mit
models:
  - LinzhouLi/RealDenseFace
short_description: Monocular 3D face reconstruction from dense UV priors
---

# RealDenseFace

Upload a portrait or select an example to reconstruct facial geometry with RealDenseFace ViT-B. Outputs are visualizations only.

[Paper](https://arxiv.org/abs/2608.09238) · [Project](https://gapszju.github.io/RealDenseFace/) · [Code](https://github.com/LinzhouLi/RealDenseFace) · [Models](https://huggingface.co/LinzhouLi/RealDenseFace)

## Deployment

This is the Space entry point. Run `python demo/hf_space/prepare.py` from the source checkout to assemble `tmp/hf-space-release`, including project sources and ten JPEG teaser inputs. Deploy with Gradio SDK and ZeroGPU hardware (`zero-a10g`).

Native FLAME solver compilation runs on CPU at startup, outside GPU callbacks. Networks are loaded once with CUDA placement at module scope; cuBLAS/cuSOLVER handles and raster contexts are created inside callbacks. Both RealDenseFace and FaceBox JIT compilation are disabled.

The standard FLAME model is a separately licensed runtime dependency and is not included in this repository. The owner's private runtime asset repository is accessed server-side using the `FLAME_READ_TOKEN` Space secret (read-only access to that repository). `FLAME_REPO_ID` and `FLAME_REVISION` Space variables select the private repository and pinned revision. Duplicated Spaces need their own licensed model and read token. Model files and credentials are never returned by the app.

Requests are serialized, inputs are limited to 1280 pixels on the longest side, and no model parameter downloads are provided. The app does not intentionally retain uploaded portraits beyond Gradio's temporary cache; temporary cached files expire after one hour.
