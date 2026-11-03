# Deployment validation

## Official Space preparation (2026-11-03)

- Official Space: `LinzhouLi/RealDenseFace`, Gradio 6.31.0, ZeroGPU, published after successful private end-to-end validation.
- Source deployment revision: `7826761c81fdfa961584aa1f2363a04fc8638fef`. Remote file listing checked: 730 source/example files, no `.pkl` or `private_assets/` paths.
- Separately licensed FLAME file is stored only in private model repository `LinzhouLi/RealDenseFace-Demo-Assets`; file revision pinned by Space variable `FLAME_REVISION=9db7fea5fccf1d7b22474c8342c39f76605c0e6a`.
- Runtime uses explicit `FLAME_READ_TOKEN` Secret only for that private asset download. Public model downloads use `token=False`. Private model cache paths are explicitly blocked from Gradio file serving.
- Source-only assembly uses a new directory `tmp/hf-space-release`, avoiding the previous private test staging directory and its asset history. The original private test Space is unchanged.
- Local syntax and Gradio UI construction checks passed; only four image components, ten JPEG examples and the simple issue link remain. The diagnostic GPU API is omitted from the official app.
- The owner configured `FLAME_READ_TOKEN` in Space Secrets. Official private runtime validation passed on examples 1 and 5, including downloading all three WebP outputs. Observed inference times were 1.65 s and 0.38 s, excluding scheduling and network; mesh overlay was visually inspected.
- Every commit in the official Space history was checked for `.pkl` and `private_assets/` paths before publication; none were present. Private asset repository visibility was verified independently.
- Switched only the official Space to public; runtime confirmed RUNNING on `zero-a10g`.
- Anonymous requests (no Authorization header) successfully loaded the public app configuration, selected a JPEG example, performed ZeroGPU reconstruction (1.69 s inference), and downloaded all three output images. An anonymous request for the private FLAME resource was denied. The original private test Space remains unchanged.

## Local baseline

- RTX 4090, Python 3.11, PyTorch 2.11.0 / CUDA 12.8.
- All ten extracted teaser inputs completed the final engine's network prediction, FOV search, FLAME fitting and rendering without errors. Warm processing time was about 0.49 seconds per image, excluding file encoding and network transfer.
- The original `fit_single_image.py --no_compile` CLI still completed successfully.

## ZeroGPU

- Private Space: https://huggingface.co/spaces/LinzhouLi/RealDenseFace-ZeroGPU-Test
- Runtime: Python 3.12.12, PyTorch 2.11.0+cu130, CUDA 13.0, Gradio 6.31.0, spaces 0.51.3.
- Hardware verified by actual GPU callback: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb.
- Native single/batched normal-equation tests matched a CPU float64 reference, maximum absolute solution error about 1.03e-7. Three separate cuBLAS/cuSOLVER/raster-context lifecycles completed successfully in one callback. Triangle rasterization covered 968 pixels on every iteration.
- Full upload → GPU reconstruction → returned visualization URLs completed successfully. The first portrait was reconstructed more than once; returned images were downloaded, decoded and visually checked.
- All ten teaser inputs passed actual cloud upload/reconstruction requests. Observed warm processing times were 0.41-0.45 s; earlier cold/warm-up requests were 1.84-2.07 s. These timings exclude client upload/download and scheduler queueing.
- The live Gradio configuration contains ten example rows. The `load_example` API successfully populated the fifth input, exercising the example-selection path.
- After image tests, two more native-probe API calls passed, each recreating and releasing contexts three times.
- The cloud's CUDA 13 compiler required replacing a cross-translation-unit templated kernel launch with a host launcher in the kernel's defining translation unit. This preserves the same kernel and arguments.
- chumpy 0.70 requires installation with build isolation disabled; it is installed before the native extension during CPU startup.

The validation loop uses authenticated REST/SSE requests. Local TLS timeouts can delay uploads and image downloads independently of GPU computation. Browser layout verification is unavailable while local browser automation is disabled.

The Space remains private. The bundled FLAME file must not be exposed by making the repository public.

## Image loading, layout and reporting update (2026-11-03)

The reporting workflow below was subsequently simplified at the user's request: the app now provides only a `Report a poor result` link, with a prefilled issue title and exactly three body sections (`Problem`, `Input image`, `Result image`). Comparison generation, description input, report session state and the report API were removed. Local Gradio construction and issue URL checks passed. Deployment commit: `36ff09b9c2dc57648a9d24415d1f17bec73ed2ab`.

- Deployed commit: `787ced5e92490775505e0061d430e6236b799c9f`; private Space confirmed RUNNING on ZeroGPU.
- All ten deployed examples are JPEG (quality 90, longest side at most 1024), approximately 1.61 MB total versus 16.70 MB for the original PNG set. No separate thumbnail assets are generated. Previous PNG examples were removed from the current deployment tree; original PPT extracts remain unchanged locally.
- Input and output columns have equal widths, matching tab headers and 420-pixel image regions. Live Gradio configuration confirms the component dimensions and ten JPEG examples. Browser automation timed out, so actual desktop/mobile visual layout has not been verified.
- Local Gradio component/event construction and feedback generation passed, including encoded Chinese descriptions and rejection when no completed reconstruction exists.
- Cloud example selection returned a 170,602-byte JPEG; retrieving it took 1.60 s in this validation run. ZeroGPU reconstruction succeeded (cold inference 2.01 s). Timings are observations on the test connection, not guarantees for users.
- A second API call in the same session generated the comparison JPEG and prefilled GitHub issue URL from the completed request's input, outputs and model/app revision. The JPEG was downloaded and visually inspected. No GitHub issue was created and no portrait was posted to GitHub.
- Temporary report output uses Gradio's normal image cache. Input changes clear prior report state and results. The issue URL contains text metadata only, with no temporary image links or credentials.
