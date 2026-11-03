"""Upload source-only code to the official ZeroGPU Space without changing visibility."""
from pathlib import Path

from huggingface_hub import HfApi

SPACE = "LinzhouLi/RealDenseFace"
ROOT = Path(__file__).resolve().parents[2]


def main():
    api = HfApi()
    info = api.space_info(SPACE)
    stage = ROOT / "tmp/hf-space-release"
    if any(stage.rglob("*.pkl")) or (stage / "private_assets").exists():
        raise RuntimeError("Refusing to upload private FLAME assets")
    if info.runtime.requested_hardware != "zero-a10g":
        raise RuntimeError("Expected ZeroGPU hardware; refusing to alter or deploy to another hardware tier")
    result = api.upload_folder(
        repo_id=SPACE, repo_type="space", folder_path=str(stage),
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        delete_patterns=["examples/example_*.png", "feedback.py"],
        commit_message="Update RealDenseFace image reconstruction demo",
    )
    print(result.commit_url)


if __name__ == "__main__":
    main()
