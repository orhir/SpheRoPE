# Copyright 2026 The SpheRoPE authors.
#
# This file is part of SpheRoPE and is licensed under the
# Creative Commons Attribution-NonCommercial 4.0 International License
# (CC BY-NC 4.0). See the LICENSE file at the repository root.

"""
LTX-2.3 ERP Panorama Pipeline — native ltx-pipelines integration.

Wraps the official TI2VidTwoStagesPipeline with ERP-specific features:
  1. Wrapped noise (spherical phase remaster on initial latent)
  2. SFC spherical RoPE (replaces width RoPE with ERP-aware embeddings)
  3. 3-way semantic distortion CFG in x0-delta space
  4. Cyclic decoding for seamless horizontal wrap-around

Uses the ltx-core / ltx-pipelines packages installed from the LTX-2 repo
(see the project README for install instructions).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import replace
from typing import Callable

import torch
from tqdm import tqdm

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer import X0Model
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoLatentShape, VideoPixelShape

from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.denoisers import (
    FactoryGuidedDenoiser,
    SimpleDenoiser,
)
from ltx_pipelines.utils.helpers import (
    combined_image_conditionings,
    get_device,
    modality_from_latent_state,
    post_process_latent,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import ModalitySpec

from ..erp_utils import (
    ERP_GUIDE_SCALE,
    apply_spherical_phase_remaster,
    circular_pad_tensor,
    circular_crop_tensor,
)

# Video-specific ERP prompt: describes both geometry AND motion behavior.
# This is the prompt appended to the user's prompt when forming the
# geometry-anchored CFG branch.
ERP_VIDEO_PROMPT = (
    "True 2:1 equirectangular projection, proper zenith/nadir pole geometry, "
    "seamless 360° horizontal wrap. "
    "Rigidly locked static tripod camera at a fixed nodal point. "
    "All geometry is permanently static: buildings, terrain, walls, floors, "
    "ceilings, roads, vegetation trunks, rocks, and all man-made structures "
    "maintain absolute pixel-locked positions across every frame. "
    "Zero structural deformation, zero background warping, zero surface drift. "
    "Only permitted motion: faint atmospheric haze, subtle light caustic shifts, "
    "microscopic dust particles. "
    "Flawless inter-frame coherence."
)

ERP_VIDEO_NEGATIVE = (
    # Quality issues
    "blurry, out of focus, overexposed, underexposed, low contrast, "
    "washed out colors, excessive noise, grainy texture, poor lighting, "
    # Structural motion
    "building movement, wall warping, ground shifting, surface drift, "
    "structural deformation, architecture bending, terrain morphing, "
    # Camera violations
    "camera movement, camera shake, panning, rotation, dolly, zoom, "
    # Temporal artifacts
    "temporal flicker, frame jump, brightness pulsing, "
    "color shifting, shadow popping, "
    # Unwanted content
    "text, watermark, logo, signature, letters, words, writing, graffiti"
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ERP RoPE Patching for native LTX-2.3 transformer
# ---------------------------------------------------------------------------

def _build_ltx23_width_angles(
    H_tokens: int,
    W_tokens: int,
    freqs_per_dim: int,
    theta: float = 10000.0,
    base_width: float = 2048.0,
    scale_w: float = 32.0,
    patch_size: int = 1,
    path_b_cycle_threshold: float = 1.0,
    r_scale: float = 1.0,
) -> torch.Tensor:
    """Build cyclic width angles for LTX2.3 with adjustable Path B threshold.

    Builds the width RoPE angles using the two-path spectral split with a
    configurable Path A/B cycle threshold.

    Args:
        path_b_cycle_threshold: Channels with fewer cycles than this use
            Path B (spherical). Default 1.0 matches the original.
        r_scale: Multiplier for the spherical radius R.
            ``R = width_span / 2 * r_scale``.  Default 1.0.

    Returns:
        (H*W, freqs_per_dim) float32 angles.
    """
    H, W = H_tokens, W_tokens
    freqs_dtype = torch.float64

    pow_indices = torch.pow(
        torch.tensor(theta, dtype=freqs_dtype),
        torch.linspace(0.0, 1.0, freqs_per_dim, dtype=freqs_dtype),
    )
    freq_vec = pow_indices * (math.pi / 2.0)

    width_span = (W * patch_size * scale_w / base_width) * 2.0
    rows = torch.arange(H, dtype=freqs_dtype)
    cols = torch.arange(W, dtype=freqs_dtype)

    # Stock positions
    _, cols_grid = torch.meshgrid(rows, cols, indexing="ij")
    scaled_w_flat = ((cols_grid.reshape(-1) * patch_size + patch_size / 2.0) * scale_w / base_width) * 2.0 - 1.0
    col_frac_flat = cols_grid.reshape(-1).to(freqs_dtype) / W

    # Stock angles
    stock_angles = scaled_w_flat.unsqueeze(1) * freq_vec.unsqueeze(0)

    # Corrected angles (Path A)
    total_angle = width_span * freq_vec
    cycles = total_angle / (2.0 * math.pi)
    rounded_cycles = torch.round(cycles)
    residual = total_angle - rounded_cycles * (2.0 * math.pi)
    corrected_angles = stock_angles - col_frac_flat.unsqueeze(1) * residual.unsqueeze(0)

    # Spherical angles (Path B)
    latitude = (rows / max(H - 1, 1)) * math.pi - (math.pi / 2.0)
    longitude = (cols / W) * 2.0 * math.pi - math.pi
    lon_grid, lat_grid = torch.meshgrid(longitude, latitude, indexing="xy")

    R_width = width_span / 2.0 * r_scale
    X_flat = (torch.cos(lat_grid) * torch.cos(lon_grid) * R_width).reshape(-1)
    Y_flat = (torch.cos(lat_grid) * torch.sin(lon_grid) * R_width).reshape(-1)

    spherical_x_angles = X_flat.unsqueeze(1) * freq_vec[0::2].unsqueeze(0)
    spherical_y_angles = Y_flat.unsqueeze(1) * freq_vec[1::2].unsqueeze(0)
    spherical_angles = torch.empty((H * W, freqs_per_dim), dtype=freqs_dtype)
    spherical_angles[:, 0::2] = spherical_x_angles
    spherical_angles[:, 1::2] = spherical_y_angles

    # Path A for channels with >= threshold cycles, Path B otherwise
    use_corrected = (cycles >= path_b_cycle_threshold)
    width_angles = torch.where(use_corrected.unsqueeze(0), corrected_angles, spherical_angles)

    return width_angles.float()


def patch_native_ltx23_rope(
    transformer_builder_fn: Callable,
    H_tokens: int,
    W_tokens: int,
    path_b_threshold: float = 1.0,
    r_scale: float = 1.0,
) -> Callable:
    """Create a patching function for the native LTX-2.3 transformer's RoPE.

    Follows the same pattern as Flux2 (SFCFluxPosEmbed) and WAN
    (SFCWanRoPEAdapter): builds the COMPLETE RoPE output from scratch
    with stock T/H axes and spherical W axis, then replaces the
    preprocessor's _prepare_positional_embeddings to return it.

    Returns a function that takes a transformer model and returns a
    restore() callable.
    """
    def patch_fn(model) -> Callable:
        """Patch the velocity_model's RoPE preprocessor."""
        # Unwrap BatchSplitAdapter if present
        actual_model = model
        if hasattr(model, '_model'):
            actual_model = model._model
        velocity_model = actual_model.velocity_model

        # Get the preprocessor
        preprocessor = velocity_model.video_args_preprocessor
        if hasattr(preprocessor, 'simple_preprocessor'):
            preprocessor = preprocessor.simple_preprocessor

        original_prepare_pe = preprocessor._prepare_positional_embeddings

        # Read config from the preprocessor
        dim = preprocessor.inner_dim
        theta = preprocessor.positional_embedding_theta
        use_double = preprocessor.double_precision_rope
        from ltx_core.model.transformer.rope import (
            LTXRopeType, generate_freq_grid_np, generate_freq_grid_pytorch,
        )
        rope_type = preprocessor.rope_type
        is_split = rope_type == LTXRopeType.SPLIT

        # Frequency vector: same as stock LTX2.3
        num_pos_dims = 3  # frame, height, width
        num_rope_elems = num_pos_dims * 2
        freqs_per_dim = dim // num_rope_elems

        freq_gen = generate_freq_grid_np if use_double else generate_freq_grid_pytorch
        freq_indices = freq_gen(theta, num_pos_dims, dim)  # (freqs_per_dim,)

        # Pre-compute cyclic width angles using Path A + Path B.
        # Path A: high-frequency channels snapped to integer harmonics (linear, uniform)
        # Path B: low-frequency channels using spherical Cartesian (inherently periodic)
        # LTX2.3 has 682 width freq channels vs Flux2's 16, so Path B needs
        # a higher cycle threshold to have meaningful influence.
        sph_width_angles = _build_ltx23_width_angles(
            H_tokens, W_tokens, freqs_per_dim, theta=theta,
            path_b_cycle_threshold=path_b_threshold,
            r_scale=r_scale,
        )  # (H*W, freqs_per_dim) float32

        logger.info(
            f"SFC RoPE patch applied: H={H_tokens}xW={W_tokens}, "
            f"freqs_per_dim={freqs_per_dim}, rope_type={rope_type}, "
            f"dim={dim}, theta={theta}"
        )

        def patched_prepare_pe(
            positions, inner_dim, max_pos, use_middle_indices_grid,
            num_attention_heads, x_dtype,
        ):
            """Build complete RoPE from scratch: stock T/H + spherical W.

            Follows the WAN/Flux2 pattern: compute each axis independently,
            combine them, then format into the expected output shape.

            Only patches the main video RoPE (3 pos dims). Cross-attention
            PE (1 pos dim) is delegated to the original function.
            """
            n_pos_dims = positions.shape[1]
            if n_pos_dims != 3 or len(max_pos) != 3:
                if not hasattr(patched_prepare_pe, '_skip_logged'):
                    logger.info(f"SFC RoPE SKIP: n_pos_dims={n_pos_dims}, max_pos={max_pos}")
                    patched_prepare_pe._skip_logged = True
                return original_prepare_pe(
                    positions, inner_dim, max_pos, use_middle_indices_grid,
                    num_attention_heads, x_dtype,
                )

            if not hasattr(patched_prepare_pe, '_call_count'):
                patched_prepare_pe._call_count = 0
            patched_prepare_pe._call_count += 1
            if patched_prepare_pe._call_count <= 3:
                logger.info(
                    f"SFC RoPE ACTIVE call #{patched_prepare_pe._call_count}: "
                    f"positions={positions.shape}, inner_dim={inner_dim}"
                )
                # Verify the output actually differs from stock
                stock_result = original_prepare_pe(
                    positions, inner_dim, max_pos, use_middle_indices_grid,
                    num_attention_heads, x_dtype,
                )
                cos_s, sin_s = stock_result

            device = positions.device
            batch_size = positions.shape[0]
            num_patches = positions.shape[2]
            num_spatial = H_tokens * W_tokens
            num_frames_t = num_patches // num_spatial

            if num_spatial == 0 or num_patches % num_spatial != 0:
                return original_prepare_pe(
                    positions, inner_dim, max_pos, use_middle_indices_grid,
                    num_attention_heads, x_dtype,
                )

            # --- Step 1: Compute midpoint positions (same as stock) ---
            if positions.ndim == 4:  # (B, 3, num_patches, 2)
                pos_start, pos_end = positions.chunk(2, dim=-1)
                midpoints = ((pos_start + pos_end) / 2.0).squeeze(-1)  # (B, 3, num_patches)
            else:
                midpoints = positions  # (B, 3, num_patches)

            # Fractional positions: midpoint / max_pos
            # (B, num_patches, 3)
            frac_pos = torch.stack(
                [midpoints[:, i] / max_pos[i] for i in range(3)], dim=-1
            ).to(device)

            # --- Step 2: Compute per-axis angles ---
            # Stock formula: angle = (frac * 2 - 1) * freq_indices[i]
            # freq_indices: (freqs_per_dim,)
            fi = freq_indices.to(device=device, dtype=torch.float32)

            # Frame axis: frac_pos[:, :, 0]
            frame_scaled = (frac_pos[:, :, 0:1] * 2 - 1)  # (B, num_patches, 1)
            frame_angles = frame_scaled * fi.unsqueeze(0)  # (B, num_patches, freqs_per_dim)

            # Height axis: frac_pos[:, :, 1]
            height_scaled = (frac_pos[:, :, 1:2] * 2 - 1)  # (B, num_patches, 1)
            height_angles = height_scaled * fi.unsqueeze(0)  # (B, num_patches, freqs_per_dim)

            # Width axis: SPHERICAL (from pre-computed angles)
            # sph_width_angles: (H*W, freqs_per_dim)
            # Tile across temporal frames
            sph_w = sph_width_angles.to(device=device, dtype=torch.float32)
            width_angles = sph_w.unsqueeze(0).expand(
                num_frames_t, -1, -1
            ).reshape(num_patches, freqs_per_dim)  # (num_patches, freqs_per_dim)
            width_angles = width_angles.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # (B, num_patches, freqs_per_dim)

            # --- Step 3: Interleave axes as [f0, h0, w0, f1, h1, w1, ...] ---
            # This matches the stock generate_freqs → transpose(-1,-2).flatten(2)
            # Stock: freqs shape after outer product is (B, num_patches, 3, freqs_per_dim)
            # After transpose(-1,-2): (B, num_patches, freqs_per_dim, 3)
            # After flatten(2): (B, num_patches, freqs_per_dim * 3)
            combined = torch.stack(
                [frame_angles, height_angles, width_angles], dim=-1
            )  # (B, num_patches, freqs_per_dim, 3)
            freqs = combined.flatten(2)  # (B, num_patches, freqs_per_dim * 3)

            # --- Step 4: Apply cos/sin and format output ---
            if is_split:
                expected_freqs = inner_dim // 2
                current_freqs = freqs.shape[-1]
                pad_size = expected_freqs - current_freqs

                cos_freq = freqs.cos()
                sin_freq = freqs.sin()

                if pad_size > 0:
                    cos_pad = torch.ones(
                        batch_size, num_patches, pad_size,
                        device=device, dtype=cos_freq.dtype
                    )
                    sin_pad = torch.zeros(
                        batch_size, num_patches, pad_size,
                        device=device, dtype=sin_freq.dtype
                    )
                    cos_freq = torch.cat([cos_pad, cos_freq], dim=-1)
                    sin_freq = torch.cat([sin_pad, sin_freq], dim=-1)

                # Reshape: (B, T, flat) → (B, T, H, D_head) → (B, H, T, D_head)
                cos_freq = cos_freq.reshape(
                    batch_size, num_patches, num_attention_heads, -1
                )
                sin_freq = sin_freq.reshape(
                    batch_size, num_patches, num_attention_heads, -1
                )
                cos_out = cos_freq.swapaxes(1, 2)
                sin_out = sin_freq.swapaxes(1, 2)
            else:
                # Interleaved mode
                cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
                sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
                pad_size = inner_dim % num_rope_elems
                if pad_size > 0:
                    cos_pad = torch.ones(
                        batch_size, num_patches, pad_size,
                        device=device, dtype=cos_freq.dtype
                    )
                    sin_pad = torch.zeros(
                        batch_size, num_patches, pad_size,
                        device=device, dtype=sin_freq.dtype
                    )
                    cos_out = torch.cat([cos_pad, cos_freq], dim=-1)
                    sin_out = torch.cat([sin_pad, sin_freq], dim=-1)
                else:
                    cos_out = cos_freq
                    sin_out = sin_freq

            result = (cos_out.to(x_dtype), sin_out.to(x_dtype))

            # Debug: verify output differs from stock on first few calls
            if hasattr(patched_prepare_pe, '_call_count') and patched_prepare_pe._call_count <= 3:
                diff = (result[0] - cos_s).abs().max().item()
                logger.info(f"  SFC RoPE output vs stock max diff: {diff:.6f}")

            return result

        preprocessor._prepare_positional_embeddings = patched_prepare_pe

        def restore():
            preprocessor._prepare_positional_embeddings = original_prepare_pe

        return restore

    return patch_fn


# ---------------------------------------------------------------------------
# ERP-aware Denoiser with 3-way semantic distortion CFG
# ---------------------------------------------------------------------------

class ERPGuidedDenoiser:
    """Denoiser with 3-way semantic distortion CFG for ERP panorama generation.

    Implements the exact 3-way CFG from the diffusers LTX2 ERP pipeline:

        x0_final = x0_cond + (gs - 1) * (x0_cond - x0_uncond)
                   + gamma * (x0_geo - x0_cond)

    This REPLACES the standard 2-way CFG (not added on top).
    Three separate forward passes: conditional, unconditional, geometric.
    """

    def __init__(
        self,
        v_context_pos: torch.Tensor,
        v_context_neg: torch.Tensor,
        v_context_geo: torch.Tensor,
        a_context_pos: torch.Tensor,
        a_context_neg: torch.Tensor,
        video_guider_factory: MultiModalGuiderFactory,
        audio_guider_factory: MultiModalGuiderFactory,
        gamma: float = ERP_GUIDE_SCALE,
        no_cfg_until_step: int = 0,
    ):
        self.v_context_pos = v_context_pos
        self.v_context_neg = v_context_neg
        self.v_context_geo = v_context_geo
        self.a_context_pos = a_context_pos
        self.a_context_neg = a_context_neg
        self.gamma = gamma
        self.no_cfg_until_step = no_cfg_until_step
        # Use the stock denoiser for steps where semantic CFG is disabled
        self._base_denoiser = FactoryGuidedDenoiser(
            v_context=v_context_pos,
            a_context=a_context_pos,
            video_guider_factory=video_guider_factory,
            audio_guider_factory=audio_guider_factory,
        )
        # Extract cfg_scale from the guider params for the 3-way formula
        # The factory builds guiders per-sigma, so we read the base params
        self._video_guider_factory = video_guider_factory
        self._audio_guider_factory = audio_guider_factory

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # For steps before no_cfg_until_step, use stock denoiser
        if step_index < self.no_cfg_until_step or self.gamma <= 0:
            return self._base_denoiser(
                transformer, video_state, audio_state, sigmas, step_index,
            )

        sigma = sigmas[step_index]

        # --- Three separate forward passes ---
        # 1. Conditional (positive prompt)
        cond_video = modality_from_latent_state(video_state, self.v_context_pos, sigma) if video_state else None
        cond_audio = modality_from_latent_state(audio_state, self.a_context_pos, sigma) if audio_state else None
        x0_cond_v, x0_cond_a = transformer(video=cond_video, audio=cond_audio, perturbations=None)

        # 2. Unconditional (negative prompt)
        uncond_video = modality_from_latent_state(video_state, self.v_context_neg, sigma) if video_state else None
        uncond_audio = modality_from_latent_state(audio_state, self.a_context_neg, sigma) if audio_state else None
        x0_uncond_v, x0_uncond_a = transformer(video=uncond_video, audio=uncond_audio, perturbations=None)

        # 3. Geometric (anchored ERP prompt)
        geo_video = modality_from_latent_state(video_state, self.v_context_geo, sigma) if video_state else None
        geo_audio = cond_audio  # reuse conditional audio context
        x0_geo_v, _ = transformer(video=geo_video, audio=geo_audio, perturbations=None)

        # --- 3-way CFG in x0-delta space (matches diffusers formula exactly) ---
        # Get the cfg_scale for this sigma from the guider factory
        sigma_val = sigma.item() if isinstance(sigma, torch.Tensor) else sigma
        video_guider = self._video_guider_factory.build_from_sigma(sigma_val)
        cfg_scale = video_guider.params.cfg_scale

        # Video: x0_final = x0_cond + (gs-1)*(x0_cond - x0_uncond) + gamma*(x0_geo - x0_cond)
        denoised_video = x0_cond_v
        if x0_cond_v is not None:
            cfg_delta = (cfg_scale - 1) * (x0_cond_v - x0_uncond_v)
            geo_delta = self.gamma * (x0_geo_v - x0_cond_v)
            denoised_video = x0_cond_v + cfg_delta + geo_delta

        # Audio: standard 2-way CFG
        audio_guider = self._audio_guider_factory.build_from_sigma(sigma_val)
        audio_cfg = audio_guider.params.cfg_scale
        denoised_audio = x0_cond_a
        if x0_cond_a is not None:
            denoised_audio = x0_cond_a + (audio_cfg - 1) * (x0_cond_a - x0_uncond_a)

        return denoised_video, denoised_audio


# ---------------------------------------------------------------------------
# Wrapped noise: apply spherical phase remaster to initial video latent
# ---------------------------------------------------------------------------

def apply_wrapped_noise_to_state(video_state: LatentState, H_tokens: int, W_tokens: int) -> LatentState:
    """Apply spherical phase remaster to the video latent in a LatentState.

    The latent is in patchified form (B, num_patches, C). We need to
    reshape to spatial form, apply per-frame remaster, then reshape back.
    """
    latent = video_state.latent  # (B, num_patches, C)
    B, num_patches, C = latent.shape
    num_spatial = H_tokens * W_tokens
    num_frames = num_patches // num_spatial

    # Reshape to (B, T, H, W, C) -> (B, C, T, H, W)
    latent_5d = latent.reshape(B, num_frames, H_tokens, W_tokens, C)
    latent_5d = latent_5d.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

    for t_idx in range(num_frames):
        latent_5d[:, :, t_idx, :, :] = apply_spherical_phase_remaster(
            latent_5d[:, :, t_idx, :, :]
        )

    # Reshape back to (B, num_patches, C)
    latent_5d = latent_5d.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
    latent = latent_5d.reshape(B, num_patches, C)

    return replace(video_state, latent=latent)


# ---------------------------------------------------------------------------
# Circular decoding wrapper
# ---------------------------------------------------------------------------

CIRCULAR_PAD_WIDTH = 4  # latent columns, matching the diffusers pipeline


def circular_decode_video(
    video_decoder: VideoDecoder,
    latent: torch.Tensor,
    tiling_config: TilingConfig | None,
    generator: torch.Generator | None,
    spatial_compression_ratio: int = 32,
) -> Iterator[torch.Tensor]:
    """Decode with circular padding for seamless horizontal wrap.

    1. Circular-pad the latent in the width dimension
    2. Decode with temporal-only tiling (spatial tiling disabled since
       circular padding handles the wrap boundary)
    3. Crop the pixel output back to original width
    """
    from ltx_core.model.video_vae import SpatialTilingConfig

    pad_w = CIRCULAR_PAD_WIDTH
    # latent shape: (B, C, T, H, W)
    latent_padded = circular_pad_tensor(latent, pad_w, dim=-1)

    # Use temporal-only tiling: keep temporal tiles for long videos,
    # but disable spatial tiling (circular padding handles width wrap,
    # and spatial tiles create visible seams).
    if tiling_config is not None:
        temporal_only_config = TilingConfig(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=32 * 1024,  # very large = effectively no spatial tiling
                tile_overlap_in_pixels=0,
            ),
            temporal_config=tiling_config.temporal_config,
        )
    else:
        temporal_only_config = None

    for chunk in video_decoder(latent_padded, tiling_config=temporal_only_config, generator=generator):
        # chunk is uint8 (F, H, W, C)
        pixel_pad = pad_w * spatial_compression_ratio
        chunk_cropped = circular_crop_tensor(chunk, pixel_pad, dim=-2)  # W is dim -2 in (F,H,W,C)
        yield chunk_cropped


# ---------------------------------------------------------------------------
# Main ERP Pipeline
# ---------------------------------------------------------------------------

class LTX23ERPPipeline:
    """Two-stage ERP panorama pipeline using native LTX-2.3 packages.

    Wraps TI2VidTwoStagesPipeline logic with ERP features:
    - Wrapped noise (spherical phase remaster)
    - SFC spherical RoPE
    - 3-way semantic distortion CFG
    - Cyclic decoding
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        multi_gpu: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        loras = loras or []

        # When multi_gpu is enabled, spread components across available GPUs
        # to avoid OOM on a single device.
        #   GPU 0: diffusion stages (transformer loaded lazily, heaviest)
        #   GPU 1: prompt encoder (Gemma 12B)
        #   GPU 2+: decoder / upsampler / audio / image conditioner
        if multi_gpu and torch.cuda.device_count() > 1:
            n_gpus = torch.cuda.device_count()
            dev_stage = torch.device("cuda:0")
            dev_encoder = torch.device("cuda:1")
            dev_decode = torch.device(f"cuda:{min(2, n_gpus - 1)}")
            logger.info(
                f"LTX23 multi-GPU: stages→{dev_stage}, encoder→{dev_encoder}, "
                f"decoder/upsampler→{dev_decode}"
            )
        else:
            dev_stage = self.device
            dev_encoder = self.device
            dev_decode = self.device

        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root, self.dtype, dev_encoder, registry=registry
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, dev_decode, registry=registry
        )
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, dev_decode, registry=registry
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path, self.dtype, dev_decode, registry=registry
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path, self.dtype, dev_decode, registry=registry
        )

        self.stage_1 = DiffusionStage(
            checkpoint_path, self.dtype, dev_stage,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path, self.dtype, dev_stage,
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
        )

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = ERP_VIDEO_NEGATIVE,
        seed: int = 42,
        height: int = 512,
        width: int = 1024,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 30,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        images: list[ImageConditioningInput] | None = None,
        output_path: str = "erp_output.mp4",
        tiling_config: TilingConfig | None = None,
        # --- ERP Parameters ---
        enable_wrapped_noise: bool = True,
        enable_sfc_rope: bool = True,
        enable_semantic_distortion_cfg: bool = True,
        enable_circular_decoding: bool = True,
        semantic_distortion_prompt: str | None = None,
        semantic_distortion_gamma: float = ERP_GUIDE_SCALE,
        no_cfg_until_step: int = 0,
        enhance_prompt: bool = False,
        path_b_threshold: float = 10.0,
        r_scale: float = 1.0,
        cfg_scale: float | None = None,
        streaming_prefetch_count: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        images = images or []
        if video_guider_params is None:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=cfg_scale or 3.0, stg_scale=1.0, rescale_scale=0.7,
                modality_scale=3.0, skip_step=0, stg_blocks=[28],
            )
        elif cfg_scale is not None:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=video_guider_params.stg_scale,
                rescale_scale=video_guider_params.rescale_scale,
                modality_scale=video_guider_params.modality_scale,
                skip_step=video_guider_params.skip_step,
                stg_blocks=video_guider_params.stg_blocks,
            )
        if audio_guider_params is None:
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                modality_scale=3.0, skip_step=0, stg_blocks=[28],
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        # --- Encode prompts ---
        prompts_to_encode = [prompt, negative_prompt]
        ctx_p, ctx_n = self.prompt_encoder(
            prompts_to_encode,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        # Move encoder outputs to the diffusion stage device (may differ in multi-GPU)
        _dev = self.device
        v_context_p = ctx_p.video_encoding.to(_dev)
        a_context_p = ctx_p.audio_encoding.to(_dev) if ctx_p.audio_encoding is not None else None
        v_context_n = ctx_n.video_encoding.to(_dev)
        a_context_n = ctx_n.audio_encoding.to(_dev) if ctx_n.audio_encoding is not None else None

        # Encode geometric prompt for 3-way CFG
        v_context_geo = None
        if enable_semantic_distortion_cfg:
            erp_prompt_text = semantic_distortion_prompt or ERP_VIDEO_PROMPT
            anchored_prompt = f"{prompt}. {erp_prompt_text}"
            ctx_geo, _ = self.prompt_encoder(
                [anchored_prompt, ""],
                streaming_prefetch_count=streaming_prefetch_count,
            )
            v_context_geo = ctx_geo.video_encoding.to(_dev)

        # --- Stage 1: Generate at half resolution ---
        stage_1_height = height // 2
        stage_1_width = width // 2
        stage_1_shape = VideoPixelShape(
            batch=1, frames=num_frames,
            width=stage_1_width, height=stage_1_height, fps=frame_rate,
        )

        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images, height=stage_1_height, width=stage_1_width,
                video_encoder=enc, dtype=dtype, device=self.device,
            )
        )

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        # Compute token dimensions for stage 1
        v_shape_s1 = VideoLatentShape.from_pixel_shape(stage_1_shape)
        # LTX2.3: spatial compression = 32, patch_size = 1
        H_tokens_s1 = v_shape_s1.height
        W_tokens_s1 = v_shape_s1.width

        logger.info(f"Stage 1: {stage_1_height}x{stage_1_width}, tokens: {H_tokens_s1}x{W_tokens_s1}")

        # Build denoiser
        if enable_semantic_distortion_cfg and v_context_geo is not None:
            stage_1_denoiser = ERPGuidedDenoiser(
                v_context_pos=v_context_p,
                v_context_neg=v_context_n,
                v_context_geo=v_context_geo,
                a_context_pos=a_context_p,
                a_context_neg=a_context_n,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params, negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params, negative_context=a_context_n,
                ),
                gamma=semantic_distortion_gamma,
                no_cfg_until_step=no_cfg_until_step,
            )
        else:
            stage_1_denoiser = FactoryGuidedDenoiser(
                v_context=v_context_p, a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params, negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params, negative_context=a_context_n,
                ),
            )

        # Build custom denoising loop with ERP features
        def erp_denoising_loop(
            sigmas, video_state, audio_state, stepper, transformer, denoiser,
        ):
            """Denoising loop with ERP RoPE patching and wrapped noise."""
            # Apply SFC RoPE patch
            restore_rope_fn = None
            if enable_sfc_rope:
                patch_fn = patch_native_ltx23_rope(None, H_tokens_s1, W_tokens_s1, path_b_threshold, r_scale=r_scale)
                restore_rope_fn = patch_fn(transformer)

            # Apply wrapped noise
            if enable_wrapped_noise and video_state is not None:
                video_state = apply_wrapped_noise_to_state(
                    video_state, H_tokens_s1, W_tokens_s1
                )

            try:
                stepper = EulerDiffusionStep()
                for step_idx, _ in enumerate(tqdm(sigmas[:-1], desc="Stage 1")):
                    denoised_video, denoised_audio = denoiser(
                        transformer, video_state, audio_state, sigmas, step_idx
                    )

                    if video_state is not None and denoised_video is not None:
                        denoised_video = post_process_latent(
                            denoised_video, video_state.denoise_mask, video_state.clean_latent
                        )
                        video_state = replace(
                            video_state,
                            latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx),
                        )

                    if audio_state is not None and denoised_audio is not None:
                        denoised_audio = post_process_latent(
                            denoised_audio, audio_state.denoise_mask, audio_state.clean_latent
                        )
                        audio_state = replace(
                            audio_state,
                            latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx),
                        )

                # Debug: check latent cyclicity after denoising
                if video_state is not None:
                    lat = video_state.latent  # (B, num_patches, C)
                    B_lat, n_patches, C_lat = lat.shape
                    n_spatial = H_tokens_s1 * W_tokens_s1
                    n_frames = n_patches // n_spatial
                    lat_5d = lat.reshape(B_lat, n_frames, H_tokens_s1, W_tokens_s1, C_lat)
                    # Check if col=0 and col=W-1 are similar
                    col0 = lat_5d[0, 0, :, 0, :]   # (H, C)
                    col1 = lat_5d[0, 0, :, 1, :]   # (H, C)
                    colWm1 = lat_5d[0, 0, :, -1, :] # (H, C)
                    d01 = (col0 - col1).abs().mean().item()
                    d0W = (col0 - colWm1).abs().mean().item()
                    d12 = (col1 - lat_5d[0, 0, :, 2, :]).abs().mean().item()
                    logger.info(
                        f"Latent cyclicity: d(col0,col1)={d01:.4f}, "
                        f"d(col0,colW-1)={d0W:.4f}, d(col1,col2)={d12:.4f}"
                    )

                return video_state, audio_state
            finally:
                if restore_rope_fn is not None:
                    restore_rope_fn()

        # Run stage 1
        video_state, audio_state = self.stage_1(
            denoiser=stage_1_denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_width,
            height=stage_1_height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=a_context_p),
            loop=erp_denoising_loop,
            streaming_prefetch_count=streaming_prefetch_count,
        )

        # --- Upsample with circular padding ---
        # Move latent to upsampler device (may differ in multi-GPU)
        _up_dev = self.upsampler._device
        s1_latent = video_state.latent[:1].to(_up_dev)  # (B, C, T, H, W)
        if enable_circular_decoding:
            # Circular-pad before upsampling so the upsampler sees wrap context
            pad_w = CIRCULAR_PAD_WIDTH
            s1_padded = circular_pad_tensor(s1_latent, pad_w, dim=-1)
            upscaled_padded = self.upsampler(s1_padded)
            # Upsampler does 2x spatial, so output pad is 2*pad_w
            upscaled_video_latent = circular_crop_tensor(upscaled_padded, pad_w * 2, dim=-1)
        else:
            upscaled_video_latent = self.upsampler(s1_latent)
        # Move back to stage device for stage 2
        upscaled_video_latent = upscaled_video_latent.to(_dev)

        # --- Stage 2: Refine at full resolution with RoPE patching ---
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images, height=height, width=width,
                video_encoder=enc, dtype=dtype, device=self.device,
            )
        )

        # Compute stage 2 token dimensions
        v_shape_s2 = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        )
        H_tokens_s2 = v_shape_s2.height
        W_tokens_s2 = v_shape_s2.width

        def erp_stage2_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser):
            """Stage 2 loop with RoPE patching."""
            restore_rope_fn = None
            if enable_sfc_rope:
                patch_fn = patch_native_ltx23_rope(None, H_tokens_s2, W_tokens_s2, path_b_threshold, r_scale=r_scale)
                restore_rope_fn = patch_fn(transformer)
            try:
                return euler_denoising_loop(
                    sigmas, video_state, audio_state, stepper, transformer, denoiser,
                )
            finally:
                if restore_rope_fn is not None:
                    restore_rope_fn()

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=distilled_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            loop=erp_stage2_loop,
            streaming_prefetch_count=streaming_prefetch_count,
        )

        # --- Decode (move latents to decoder device for multi-GPU) ---
        _dec_dev = self.video_decoder._device
        decode_video_latent = video_state.latent.to(_dec_dev)
        decode_audio_latent = audio_state.latent.to(_dec_dev)
        if enable_circular_decoding:
            decoded_video = circular_decode_video(
                self.video_decoder, decode_video_latent,
                tiling_config, generator,
            )
        else:
            decoded_video = self.video_decoder(decode_video_latent, tiling_config, generator)

        decoded_audio = self.audio_decoder(decode_audio_latent)
        return decoded_video, decoded_audio


