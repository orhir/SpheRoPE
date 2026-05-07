#!/usr/bin/env python
"""
Generate a 360° equirectangular panorama from a text prompt using
ERP-enhanced FLUX.1, FLUX.2, or LTX-2.3 pipelines.

Usage::

    # FLUX.1 (image)
    python generate_panorama.py --model flux1 --prompt "sunlit forest clearing" --output pano.png

    # FLUX.2 (image, highest quality)
    python generate_panorama.py --model flux2 --prompt "..." --output pano.png

    # LTX-2.3 (video panorama, requires LTX-2 repo installed locally — see README)
    python generate_panorama.py --model ltx23 --prompt "..." --output pano.mp4 \\
        --ltx23-checkpoint-dir /path/to/ltx23_checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_RES = {
    "flux1": (512, 1024),
    "flux2": (1024, 2048),
    "ltx23": (1024, 2048),
}
DEFAULT_STEPS = {"flux1": 28, "flux2": 50, "ltx23": 30}
DEFAULT_GUIDANCE = {"flux1": 3.5, "flux2": 4.0}  # ltx23 uses its own cfg_scale


def _resolve_ltx23_lora(ckpt_dir: Path) -> Path:
    """Locate the LTX-2.3 distilled LoRA in ``ckpt_dir``.

    Lightricks has shipped this file under at least two names:

    - ``ltx-2.3-22b-distilled-lora-384-1.1.safetensors``  (current HF name)
    - ``ltx-2.3-22b-distilled-lora-384.safetensors``      (earlier releases)

    We try them in order and raise a readable error if neither exists.
    """
    candidates = [
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "ltx-2.3-22b-distilled-lora-384.safetensors",
    ]
    for name in candidates:
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find an LTX-2.3 distilled LoRA in {ckpt_dir}.\n"
        f"Expected one of: {candidates}\n"
        "Download it from https://huggingface.co/Lightricks/LTX-2.3-22b-distilled-lora-384"
    )


def _run_flux(args: argparse.Namespace) -> None:
    """Generate an ERP panorama with FLUX.1 or FLUX.2."""
    from panorama_diffusers import ERPFlux2Pipeline, ERPFluxPipeline

    if args.model == "flux1":
        pipe_cls = ERPFluxPipeline
        model_id = args.model_id or "black-forest-labs/FLUX.1-dev"
    else:
        pipe_cls = ERPFlux2Pipeline
        model_id = args.model_id or "black-forest-labs/FLUX.2-dev"

    height, width = args.height or DEFAULT_RES[args.model][0], args.width or DEFAULT_RES[args.model][1]
    pipe = pipe_cls.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps or DEFAULT_STEPS[args.model],
        guidance_scale=args.guidance_scale or DEFAULT_GUIDANCE[args.model],
        generator=generator,
        # --- ERP features ---
        sfc_rope=True,
        enable_circular_encoding=True,
        semantic_distortion_cfg=True,
        semantic_distortion_gamma=args.erp_gamma,
        no_cfg_until_timestep=args.no_cfg_until_timestep,
    ).images[0]

    image.save(args.output)
    print(f"Saved panorama → {args.output}")


def _run_ltx23(args: argparse.Namespace) -> None:
    """Generate an ERP panorama video with LTX-2.3."""
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig
    from ltx_pipelines.utils.media_io import encode_video

    from panorama_diffusers.pipelines.pipeline_ltx23_erp import LTX23ERPPipeline

    ckpt = Path(args.ltx23_checkpoint_dir)
    distilled_lora_path = _resolve_ltx23_lora(ckpt)

    pipe = LTX23ERPPipeline(
        checkpoint_path=str(ckpt / "ltx-2.3-22b-dev.safetensors"),
        distilled_lora=[LoraPathStrengthAndSDOps(
            str(distilled_lora_path),
            0.8, LTXV_LORA_COMFY_RENAMING_MAP,
        )],
        spatial_upsampler_path=str(ckpt / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        gemma_root=str(ckpt / "gemma-3-12b-it-qat-q4_0-unquantized"),
        multi_gpu=args.multi_gpu,
    )

    height, width = args.height or DEFAULT_RES["ltx23"][0], args.width or DEFAULT_RES["ltx23"][1]
    video_iter, audio = pipe(
        prompt=args.prompt,
        seed=args.seed,
        height=height,
        width=width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps or DEFAULT_STEPS["ltx23"],
        tiling_config=TilingConfig.default(),
        enable_wrapped_noise=False,
        enable_sfc_rope=True,
        enable_circular_decoding=True,
        enable_semantic_distortion_cfg=True,
        semantic_distortion_gamma=args.erp_gamma,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video_iter, fps=24.0, audio=audio,
        output_path=str(out_path), video_chunks_number=1,
    )
    print(f"Saved panorama video → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a 360° ERP panorama from a text prompt.")
    p.add_argument("--model", choices=["flux1", "flux2", "ltx23"], required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True, help="Output file (.png for flux, .mp4 for ltx23).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None, help="FLUX only.")
    p.add_argument("--erp-gamma", type=float, default=6.0,
                   help="Semantic distortion CFG strength (default: 6.0; LTX-2.3 recommended: 3.0).")
    p.add_argument("--no-cfg-until-timestep", type=int, default=2, help="FLUX only.")
    p.add_argument("--model-id", type=str, default=None, help="Override HuggingFace model id (FLUX).")
    p.add_argument("--num-frames", type=int, default=121, help="LTX-2.3 only.")
    p.add_argument("--ltx23-checkpoint-dir", type=str, default=None,
                   help="Directory containing LTX-2.3 checkpoints (required for --model ltx23).")
    p.add_argument("--multi-gpu", action="store_true", help="LTX-2.3 only: spread components across GPUs.")
    p.add_argument("--offload", choices=["model", "sequential"], default="model",
                   help="CPU offload strategy (FLUX only). 'model' is faster but uses more VRAM (~16 GB peak); "
                        "'sequential' is slower but runs on ~8 GB VRAM. Default: model.")
    args = p.parse_args()

    if args.model == "ltx23":
        if not args.ltx23_checkpoint_dir:
            p.error("--ltx23-checkpoint-dir is required when --model ltx23")
        _run_ltx23(args)
    else:
        _run_flux(args)


if __name__ == "__main__":
    main()
