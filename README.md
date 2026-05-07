<a href="https://orhir.github.io/SpheRoPE/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg"></a>

# SpheRoPE: Zero-Shot Optimization-Free 360° Panorama Generation with Spherical RoPE [Paper TBD]

A PyTorch implementation of a zero-shot, optimization-free method for
generating seamless 360° equirectangular panoramas with pre-trained
diffusion models, by rewiring their positional encoding and
classifier-free guidance to respect spherical geometry.

## Overview

SpheRoPE adapts stock text-to-image and text-to-video diffusion models
to produce valid equirectangular panoramas *without* any fine-tuning or
per-image optimization. The method plugs three lightweight modifications
into the sampling loop:

1. **SFC spherical RoPE** — replaces the width axis of the model's
   rotary positional encoding with a sphere-aware variant, so attention
   respects the 360° horizontal wrap and pole convergence.
2. **Semantic-distortion CFG** — a 3-way classifier-free-guidance in
   which a geometry-anchored prompt pulls denoising toward a valid ERP
   layout.
3. **Circular encoding / decoding** — pads latents circularly along the
   width axis so the ±180° seam is pixel-continuous.

Together these give coherent, seam-free panoramas from pre-trained
FLUX.1, FLUX.2, and LTX-2.3 with no extra training data and no
optimization loop at inference.

### Supported models

| Model    | Type  | Resolution       | Pipeline              |
|----------|-------|------------------|-----------------------|
| FLUX.1-dev | Image | 512 × 1024       | `ERPFluxPipeline`     |
| FLUX.2-dev | Image | 1024 × 2048      | `ERPFlux2Pipeline`    |
| LTX-2.3    | Video | 1024 × 2048, 121 f | `LTX23ERPPipeline`  |


## Install

First, install a PyTorch build that matches your CUDA driver.  The pip
default pulls the latest PyTorch which currently ships with CUDA 13 and
will not work on machines still on CUDA 12 drivers — so install torch
explicitly first:

```bash
# Example for CUDA 12.x drivers (check your driver with `nvidia-smi`):
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Then install the rest:
git clone https://github.com/orhir/SpheRoPE.git
cd SpheRoPE
pip install -r requirements.txt
pip install -e .
```

### Additional setup for LTX-2.3 (video)

LTX-2.3 is not on PyPI. Install it from the official repository in the
same Python environment:

```bash
# 1. Clone and install Lightricks' LTX-2
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
pip install uv
uv sync --frozen
source .venv/bin/activate

# uv does not ship a modern `pip` inside the new venv, so seed one:
python -m ensurepip --upgrade

# 2. Install SpheRoPE into the LTX venv
cd ../SpheRoPE
python -m pip install -r requirements.txt
python -m pip install -e .

# 3. Download checkpoints from https://huggingface.co/Lightricks
#    into a single directory, e.g. ./ltx23_checkpoints/
#      - ltx-2.3-22b-dev.safetensors                    (main model)
#      - ltx-2.3-22b-distilled-lora-384-1.1.safetensors (distilled LoRA)
#      - ltx-2.3-spatial-upscaler-x2-1.1.safetensors    (spatial upscaler)
#      - gemma-3-12b-it-qat-q4_0-unquantized/           (text encoder, full dir)
```

The FLUX pipelines only require `pip install -r requirements.txt`;
`ltx_core` / `ltx_pipelines` are *not* needed unless you use `--model ltx23`.

## Usage

### FLUX.1 (image, fastest)

```bash
python generate_panorama.py --model flux1 \
    --prompt "sunlit forest clearing, wildflowers, dappled light" \
    --output pano_flux1.png
```

### FLUX.2 (image, highest quality)

```bash
python generate_panorama.py --model flux2 \
    --prompt "a cozy library with leather armchairs and tall wooden bookshelves" \
    --output pano_flux2.png
```

### LTX-2.3 (video)

```bash
python generate_panorama.py --model ltx23 \
    --prompt "a cozy library with leather armchairs, gentle light flickering through the windows" \
    --output pano_ltx23.mp4 \
    --ltx23-checkpoint-dir ./ltx23_checkpoints \
    --num-frames 121
```

### Python API

```python
import torch
from panorama_diffusers import ERPFlux2Pipeline

pipe = ERPFlux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="a marble cathedral interior, sunbeams through stained glass",
    height=1024, width=2048,
    num_inference_steps=50, guidance_scale=4.0,
    # --- ERP features ---
    sfc_rope=True,
    enable_circular_encoding=True,
    semantic_distortion_cfg=True,
    semantic_distortion_gamma=6.0,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("pano.png")
```

### Common options

| Flag                      | Default | Notes |
|---------------------------|---------|-------|
| `--height / --width`      | model   | Must keep 1:2 aspect ratio for a proper ERP. |
| `--num-inference-steps`   | 28/50/30 | FLUX.1 / FLUX.2 / LTX-2.3 |
| `--guidance-scale`        | 3.5/4.0 | FLUX only; CFG on the base prompt. |
| `--erp-gamma`             | 6.0     | Strength of the ERP geometric nudge. LTX-2.3 is tuned for ~3.0. |
| `--no-cfg-until-timestep` | 2       | FLUX only; skip ERP CFG for the first N denoising steps. |
| `--offload`               | model   | FLUX only; `model` (fast, ~16 GB VRAM) or `sequential` (slow, ~8 GB VRAM). |

## Memory

Peak VRAM requirements:

| Model    | With `--offload model` | With `--offload sequential` | Tested on |
|----------|------------------------|------------------------------|-----------|
| FLUX.1-dev  | ~16 GB | ~8 GB  | 1×A100-40G |
| FLUX.2-dev  | ~40 GB | ~16 GB | 1×A100-80G |
| LTX-2.3     | ~70 GB (single) / ~25 GB per GPU with `--multi-gpu` | N/A | 1×A100-80G or multi-GPU |

Sequential offload is slower (FLUX.1: ~3× slower; FLUX.2: ~10× slower,
about 20 minutes per image at 1024×2048/50 steps on an A100) but fits on
consumer GPUs.  Prefer `--offload model` whenever your GPU has enough
VRAM.


## Evaluation

Instructions for downloading the ODISR benchmark and generating
Qwen3-VL captions used for quantitative evaluation are in
[EVAL.md](./EVAL.md).

## License

This repository is released under the
[Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0)](./LICENSE).

FLUX.1-dev, FLUX.2-dev and LTX-2.3 model weights are governed by their
own respective licenses (see the Black Forest Labs and Lightricks
HuggingFace model pages).

## Acknowledgements

This work is built on top of
[🤗 diffusers](https://github.com/huggingface/diffusers) (Apache 2.0),
[FLUX.1 / FLUX.2](https://github.com/black-forest-labs/flux) (non-commercial),
and [LTX-2.3](https://github.com/Lightricks/LTX-2).

