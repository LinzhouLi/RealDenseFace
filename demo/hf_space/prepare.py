"""Assemble the public-ready Space sources, excluding private runtime assets."""
from pathlib import Path
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
DEST = ROOT / "tmp/hf-space-release"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "feedback.py").unlink(missing_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyd", "*.so", "build", "*.egg-info", ".git")
    for name in ("camera", "common", "configs", "models", "tracker", "visualization", "cuda_extensions"):
        shutil.copytree(ROOT / name, DEST / name, dirs_exist_ok=True, ignore=ignore)
    for name in ("README.md", "requirements.txt", "packages.txt", "app.py", "engine.py", "bootstrap.py", "diagnostics.py"):
        shutil.copy2(SOURCE / name, DEST / name)
    shutil.copy2(ROOT / "LICENSE", DEST / "LICENSE")
    (DEST / "examples").mkdir(exist_ok=True)
    examples = sorted((ROOT / "tmp/demo-feasibility").glob("example_[0-9][0-9].png"))
    if len(examples) != 10:
        raise RuntimeError(f"Expected ten extracted teaser examples, found {len(examples)}")
    for image in examples:
        with Image.open(image) as source:
            source = source.convert("RGB")
            source.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            source.save(DEST / "examples" / (image.stem + ".jpg"), quality=90, optimize=True)
    # Remove only the previous generated examples, preserving original PPT extracts.
    for image in (DEST / "examples").glob("example_[0-9][0-9].png"):
        image.unlink()
    if any(DEST.rglob("*.pkl")) or (DEST / "private_assets").exists():
        raise RuntimeError("Release directory must not contain private FLAME assets")
    print(f"Assembled public-ready sources at {DEST}; private FLAME assets excluded.")


if __name__ == "__main__":
    main()
