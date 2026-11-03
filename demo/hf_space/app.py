import spaces  # Must precede torch and CUDA-touching project imports.

import logging
import os
from pathlib import Path
import time
from urllib.parse import urlencode

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch
from huggingface_hub import hf_hub_download

from bootstrap import prepare_solver

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "0a74d1d328ed6f47da5ec3632991c02a0266adca"
prepare_solver()

from engine import ReconstructionEngine


def weight(name: str) -> str:
    return hf_hub_download("LinzhouLi/RealDenseFace", filename=name, revision=MODEL_REVISION, token=False)


def flame_model() -> str:
    token = os.environ.get("FLAME_READ_TOKEN")
    if not token:
        raise RuntimeError("Configure the FLAME_READ_TOKEN Space secret with read access to the private demo assets repository.")
    return hf_hub_download(
        os.environ.get("FLAME_REPO_ID", "LinzhouLi/RealDenseFace-Demo-Assets"),
        filename="flame2023.pkl", revision=os.environ.get("FLAME_REVISION", "main"),
        token=token,
    )


print("[startup] Loading ViT-B and FaceBox", flush=True)
FLAME_MODEL_PATH = flame_model()
ENGINE = ReconstructionEngine(
    ROOT, weight("weights/realdenseface/vitb.pth"), weight("weights/facebox/face_box.pth"),
    weight("weights/flame/flame_assets.npz"), FLAME_MODEL_PATH,
)
print("[startup] Networks loaded; native contexts will be created in GPU requests", flush=True)


@spaces.GPU(duration=20)
def reconstruct(image: Image.Image):
    """Reconstruct a single portrait and return dense priors, overlay and mesh visualizations."""
    started = time.perf_counter()
    if image is None:
        raise gr.Error("Upload a portrait or choose an example first.")
    if image.width * image.height > 40_000_000:
        raise gr.Error("Please use an image with fewer than 40 million pixels.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    if min(image.size) < 32:
        raise gr.Error("The image is too small. Please upload a larger portrait.")
    try:
        dense, overlay, mesh, report = ENGINE.reconstruct(np.ascontiguousarray(np.asarray(image)))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception:
        logging.exception("Reconstruction failed")
        raise gr.Error("Reconstruction failed. Please try a clearer portrait.")
    report["callback_seconds"] = round(time.perf_counter() - started, 3)
    print(f"[request] {report}", flush=True)
    return dense, overlay, mesh, f"Inference: {report['seconds']:.2f} s (excludes queue and transfer)."



with gr.Blocks(title="RealDenseFace", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("""# RealDenseFace
### Real-time Monocular 3D Face Reconstruction from Dense UV-space Priors
[Paper](https://arxiv.org/abs/2608.09238) · [Project](https://gapszju.github.io/RealDenseFace/) · [Code](https://github.com/LinzhouLi/RealDenseFace) · [Models](https://huggingface.co/LinzhouLi/RealDenseFace)

Upload a portrait or select an example, then click **Reconstruct**. Powered by RealDenseFace ViT-B.
""")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            with gr.Tabs():
                with gr.Tab("Input"):
                    image_input = gr.Image(type="pil", label="Input portrait", sources=["upload"], height=420, format="webp")
        with gr.Column(scale=1, min_width=300):
            with gr.Tabs():
                with gr.Tab("Mesh overlay"):
                    overlay = gr.Image(label="Mesh overlay", interactive=False, height=420, format="webp")
                with gr.Tab("Reconstructed mesh"):
                    mesh = gr.Image(label="Reconstructed mesh", interactive=False, height=420, format="webp")
                with gr.Tab("Dense priors"):
                    dense = gr.Image(label="Dense priors", interactive=False, height=420, format="webp")
    with gr.Row():
        run = gr.Button("Reconstruct", variant="primary", scale=1)
        status = gr.Markdown(scale=1)
    run.click(reconstruct, inputs=image_input, outputs=[dense, overlay, mesh, status],
              api_name="reconstruct", concurrency_limit=1, concurrency_id="gpu")
    gr.Examples(examples=[[str(p)] for p in sorted((ROOT / "examples").glob("*.jpg"))],
                inputs=[image_input], label="Examples", examples_per_page=10, cache_examples=False)
    gr.Markdown("Please use an image with one clearly visible face. ZeroGPU requests may queue before processing.")
    issue_url = "https://github.com/LinzhouLi/RealDenseFace/issues/new?" + urlencode({
        "title": "[Demo] Poor reconstruction result",
        "body": "### Problem\nPlease describe the problem.\n\n"
                "### Input image\nUpload your input image here.\n\n"
                "### Result image\nUpload your result image here.\n",
    })
    gr.Markdown(f"[Report a poor result]({issue_url})")
    image_input.change(lambda: (None, None, None, ""),
                       inputs=[], outputs=[dense, overlay, mesh, status],
                       queue=False, api_name=False)


if __name__ == "__main__":
    demo.queue(max_size=12).launch(show_error=False, blocked_paths=[FLAME_MODEL_PATH, str(Path(FLAME_MODEL_PATH).resolve())])
