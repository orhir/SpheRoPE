# Copyright 2026 The SpheRoPE authors.
#
# This file is part of SpheRoPE and is licensed under the
# Creative Commons Attribution-NonCommercial 4.0 International License
# (CC BY-NC 4.0). See the LICENSE file at the repository root.

"""
Shared ERP (Equirectangular Projection) panorama utilities.

Utility functions shared across the FLUX.1-dev, FLUX.2-dev, and LTX-2.3
ERP panorama pipelines.

Provides:
- Constants: ERP_PROMPT, ERP_GUIDE_SCALE
- Circular encoding helpers: circular_pad_tensor, circular_crop_tensor,
  decoder_circular_pad_width
- SFC Spherical RoPE: build_sfc_spherical_rope (FLUX.1),
  build_sfc_spherical_rope_flux2 (FLUX.2), SFCFluxPosEmbed wrapper
- Noise wrapping: apply_spherical_phase_remaster
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ERP_PROMPT is used by the ERP pipelines as the geometry-anchored prompt for
# 3-way semantic-distortion CFG.  It describes, in natural language, what a
# valid equirectangular projection looks like.
ERP_PROMPT = (
    "Single unified continuous environment, monolithic scene composition, "
    "solitary spatial layout, flawlessly stitched 360 panorama, true "
    "equirectangular projection, accurate spherical geometry, continuous "
    "horizontal wrap, zero parallax error."
)


ERP_GUIDE_SCALE = 6.0  # Default semantic_distortion_gamma


# ---------------------------------------------------------------------------
# Circular Encoding Helpers
# ---------------------------------------------------------------------------

def circular_pad_tensor(tensor: torch.Tensor, pad_w: int, dim: int = -1) -> torch.Tensor:
    """Circularly pad along the given spatial dimension.

    Borrows `pad_w` elements from the end and prepends them, and `pad_w`
    elements from the start and appends them, simulating horizontal
    wrap-around for ERP panoramas.

    Args:
        tensor: Input tensor of any shape.
        pad_w: Number of elements to pad on each side.
        dim: Dimension along which to pad (default: -1, i.e. last dim).

    Returns:
        Padded tensor with `tensor.shape[dim] + 2 * pad_w` along `dim`.
    """
    if pad_w == 0:
        return tensor
    left = tensor.narrow(dim, tensor.shape[dim] - pad_w, pad_w)
    right = tensor.narrow(dim, 0, pad_w)
    return torch.cat([left, tensor, right], dim=dim)


def circular_crop_tensor(tensor: torch.Tensor, pad_w: int, dim: int = -1) -> torch.Tensor:
    """Crop `pad_w` elements from each side along the given dimension.

    This is the inverse of :func:`circular_pad_tensor`. After a circular
    pad → process → crop round-trip the tensor is restored to its
    original width.

    Args:
        tensor: Input tensor (typically the output of a padded operation).
        pad_w: Number of elements to crop from each side.
        dim: Dimension along which to crop (default: -1, i.e. last dim).

    Returns:
        Cropped tensor with `tensor.shape[dim] - 2 * pad_w` along `dim`.
    """
    if pad_w == 0:
        return tensor
    return tensor.narrow(dim, pad_w, tensor.shape[dim] - 2 * pad_w)


def decoder_circular_pad_width(receptive_field: int = 18) -> int:
    """Return one-side circular padding width in latent columns for VAE decoder.

    The default value of 18 matches the measured one-sided receptive field
    of the FLUX VAE decoder, ensuring every output pixel has full circular
    context at the horizontal wrap boundary.

    Args:
        receptive_field: Decoder receptive field in latent columns.

    Returns:
        One-side padding width (same as ``receptive_field``).
    """
    return receptive_field


# ---------------------------------------------------------------------------
# SFC Spherical RoPE
# ---------------------------------------------------------------------------

def build_sfc_spherical_rope(
    H_tokens: int,
    W_tokens: int,
    dim: int = 128,
    max_error: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build explicit (cos, sin) RoPE tensors with spherical ERP geometry.

    Constructs rotary position embeddings that understand equirectangular
    projection geometry. The 128-dim output is structured as
    ``[TIME(16) | HEIGHT(56) | WIDTH(56)]``, where TIME and HEIGHT are
    identical to stock Flux RoPE and WIDTH uses a two-path spectral split:

    - **Path A** (high-frequency, harmonically quantized): channels whose
      RoPE frequency aligns with an integer harmonic of the fundamental
      ``2π / W_tokens`` use quantized linear positions, guaranteeing exact
      horizontal wrap-around periodicity.
    - **Path B** (low-frequency, spherical 3D): remaining channels use
      Cartesian coordinates on the unit sphere (X on even slots, Y on odd
      slots). At the poles ``cos(±π/2) = 0``, so all polar tokens share
      identical width embeddings.

    All intermediate computations use ``float64`` for numerical precision;
    the final output is cast to ``float32``.

    Args:
        H_tokens: Number of token rows (height).
        W_tokens: Number of token columns (width).
        dim: Total RoPE dimensionality (default: 128).
        max_error: Maximum allowed quantization error for harmonic
            classification (default: 0.06).

    Returns:
        Tuple ``(cos_emb, sin_emb)`` each of shape
        ``(1, H_tokens * W_tokens, dim)``, dtype ``float32``.
    """
    base = 10000.0
    H, W = H_tokens, W_tokens

    rows = torch.arange(H, dtype=torch.float64)
    cols = torch.arange(W, dtype=torch.float64)

    rows_grid, cols_grid = torch.meshgrid(rows, cols, indexing="ij")
    Z_flat = rows_grid.reshape(-1)
    W_flat = cols_grid.reshape(-1)

    # Closed-interval latitude: row 0 → south pole, row H-1 → north pole
    theta = (rows / (H - 1)) * math.pi - (math.pi / 2.0)
    phi = (cols / W) * 2.0 * math.pi - math.pi
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="xy")

    R_width = W / (2.0 * math.pi)
    X_spherical = torch.cos(theta_grid) * torch.cos(phi_grid) * R_width
    Y_spherical = torch.cos(theta_grid) * torch.sin(phi_grid) * R_width
    X_flat = X_spherical.reshape(-1)
    Y_flat = Y_spherical.reshape(-1)

    # TIME: 8 frequencies → 16 dims (constant=1 for all tokens)
    time_dim = 16
    time_freqs = 1.0 / (base ** (2.0 * torch.arange(time_dim // 2, dtype=torch.float64) / time_dim))
    time_angles = torch.ones(H * W, 1, dtype=torch.float64) * time_freqs.unsqueeze(0)

    # HEIGHT: 28 frequencies → 56 dims (linear row index, same as stock FLUX)
    height_dim = 56
    height_freqs = 1.0 / (base ** (2.0 * torch.arange(height_dim // 2, dtype=torch.float64) / height_dim))
    height_angles = Z_flat.unsqueeze(1) * height_freqs.unsqueeze(0)

    # WIDTH: 28 frequencies → 56 dims (spectral split)
    width_dim = 56
    width_all_freqs = 1.0 / (base ** (2.0 * torch.arange(width_dim // 2, dtype=torch.float64) / width_dim))

    fundamental_freq = (2.0 * math.pi) / W
    k_values = width_all_freqs / fundamental_freq
    rounded_k = torch.round(k_values)
    quantization_error = torch.abs(k_values - rounded_k) / (k_values + 1e-8)
    valid_mask = (k_values >= 1.0) & (quantization_error <= max_error)

    invalid_indices = torch.where(~valid_mask)[0]
    split_idx = invalid_indices[0].item() if len(invalid_indices) > 0 else len(width_all_freqs)
    step_mask = torch.arange(len(width_all_freqs), device=X_flat.device) < split_idx

    # Path A: harmonically quantized linear angles
    harmonic_freqs = rounded_k * fundamental_freq
    linear_angles = W_flat.unsqueeze(1) * harmonic_freqs.unsqueeze(0)

    # Path B: spherical 3D angles (X on even slots, Y on odd slots)
    spherical_x_angles = X_flat.unsqueeze(1) * width_all_freqs[0::2].unsqueeze(0)
    spherical_y_angles = Y_flat.unsqueeze(1) * width_all_freqs[1::2].unsqueeze(0)
    spherical_angles = torch.empty((H * W, 28), dtype=torch.float64, device=X_flat.device)
    spherical_angles[:, 0::2] = spherical_x_angles
    spherical_angles[:, 1::2] = spherical_y_angles

    width_angles = torch.where(step_mask.unsqueeze(0), linear_angles, spherical_angles)

    # Combine: [TIME(8) | HEIGHT(28) | WIDTH(28)] = 64 half-angles
    half_angles = torch.cat([time_angles, height_angles, width_angles], dim=-1)

    # Duplicate each dim via repeat_interleave to get 128-dim output
    cos_emb = torch.cos(half_angles).repeat_interleave(2, dim=-1).float().unsqueeze(0)
    sin_emb = torch.sin(half_angles).repeat_interleave(2, dim=-1).float().unsqueeze(0)

    return cos_emb, sin_emb


def build_sfc_spherical_rope_flux2(
    H_tokens: int,
    W_tokens: int,
    axes_dims: tuple[int, ...] = (32, 32, 32, 32),
    theta: float = 2000.0,
    max_error: float = 0.06,
    scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build explicit (cos, sin) RoPE tensors with spherical ERP geometry for FLUX.2.

    FLUX.2 uses a four-axis RoPE ``[T | H | W | L]`` (default 32 dims each,
    total 128).  This function replaces only the ``W`` (width) axis with a
    two-path spectral split:

    - **Path A** (NTK-aware scaled harmonic): low channel indices whose RoPE
      frequency aligns with an integer harmonic of ``2π / W_tokens`` are
      snapped to an NTK-scaled requantized harmonic, preserving exact
      horizontal periodicity.
    - **Path B** (spherical 3-D): remaining channels use Cartesian ``(X, Y)``
      coordinates on the unit sphere, scaled by ``W / scale`` and shifted to
      be non-negative.  At the poles ``cos(±π/2) = 0``, so all polar tokens
      share identical width embeddings, encoding pole convergence.

    ``T`` (time) and ``L`` (level) axes receive all-zero positions (constant
    per token) and ``H`` (height) uses the stock linear row index.

    All intermediate computations use ``float64`` for numerical precision;
    the final output is cast to ``float32``.

    Args:
        H_tokens: Number of token rows (height).
        W_tokens: Number of token columns (width).
        axes_dims: Per-axis dimensionality ``(T, H, W, L)``.  Default
            ``(32, 32, 32, 32)`` matches FLUX.2.
        theta: RoPE base (default: 2000.0, matches FLUX.2).
        max_error: Maximum allowed quantization error for harmonic
            classification (default: 0.06).
        scale: Path A NTK scaling factor and Path B radius denominator.

    Returns:
        Tuple ``(cos_emb, sin_emb)`` each of shape
        ``(1, H_tokens * W_tokens, sum(axes_dims))``, dtype ``float32``.
    """
    H, W = H_tokens, W_tokens
    t_dim, h_dim, w_dim, l_dim = axes_dims

    rows = torch.arange(H, dtype=torch.float64)
    cols = torch.arange(W, dtype=torch.float64)

    rows_grid, cols_grid = torch.meshgrid(rows, cols, indexing="ij")
    Z_flat = rows_grid.reshape(-1)  # [H*W] — height positions
    W_flat = cols_grid.reshape(-1)  # [H*W] — width positions

    # Spherical coordinates for width replacement
    theta_lat = (rows / max(H - 1, 1)) * math.pi - (math.pi / 2.0)
    phi_lon = (cols / W) * 2.0 * math.pi - math.pi
    phi_grid, theta_grid = torch.meshgrid(phi_lon, theta_lat, indexing="xy")

    # R_width = W / scale
    R_width = W / scale
    X_spherical = (torch.cos(theta_grid) * torch.cos(phi_grid) + 1.0) * R_width
    Y_spherical = (torch.cos(theta_grid) * torch.sin(phi_grid) + 1.0) * R_width
    X_flat = X_spherical.reshape(-1)
    Y_flat = Y_spherical.reshape(-1)

    # Helper: compute 1D RoPE half-angles for a given axis dim and positions
    def _axis_half_angles(dim: int, positions: torch.Tensor) -> torch.Tensor:
        """Return half-angles of shape (H*W, dim//2)."""
        half = dim // 2
        freqs = 1.0 / (theta ** (2.0 * torch.arange(half, dtype=torch.float64) / dim))
        return positions.unsqueeze(1) * freqs.unsqueeze(0)

    # T axis: all-zero positions (constant for image tokens)
    t_angles = _axis_half_angles(t_dim, torch.zeros(H * W, dtype=torch.float64))

    # H axis: linear row index (same as stock Flux 2)
    h_angles = _axis_half_angles(h_dim, Z_flat)

    # W axis: NTK-aware scaled frequencies
    w_half = w_dim // 2
    w_freqs = 1.0 / (theta ** (2.0 * torch.arange(w_half, dtype=torch.float64) / w_dim))

    fundamental_freq = (2.0 * math.pi) / W
    k_values = w_freqs / fundamental_freq
    rounded_k = torch.round(k_values)
    quantization_error = torch.abs(k_values - rounded_k) / (k_values + 1e-8)
    valid_mask = (k_values >= 1.0) & (quantization_error <= max_error)

    invalid_indices = torch.where(~valid_mask)[0]
    split_idx = invalid_indices[0].item() if len(invalid_indices) > 0 else len(w_freqs)
    step_mask = torch.arange(len(w_freqs)) < split_idx


    # Path A: NTK-aware scaled harmonic angles
    harmonic_freqs = rounded_k * fundamental_freq

    n_path_a = split_idx
    if n_path_a > 1:
        a_indices = torch.arange(n_path_a, dtype=torch.float64)
        ntk_scale = scale ** (a_indices / (n_path_a - 1))
    else:
        ntk_scale = torch.ones(max(n_path_a, 1), dtype=torch.float64)

    # Scale frequencies and re-quantize to preserve cyclicity
    scaled_freqs = harmonic_freqs[:split_idx] / ntk_scale[:split_idx]
    scaled_k = scaled_freqs / fundamental_freq
    requantized_k = torch.round(scaled_k).clamp(min=1.0)
    requantized_freqs = requantized_k * fundamental_freq

    linear_angles = W_flat.unsqueeze(1) * harmonic_freqs.unsqueeze(0)
    linear_angles[:, :split_idx] = W_flat.unsqueeze(1) * requantized_freqs.unsqueeze(0)

    # Path B: spherical 3D angles (X on even slots, Y on odd slots)
    spherical_x_angles = X_flat.unsqueeze(1) * w_freqs[0::2].unsqueeze(0)
    spherical_y_angles = Y_flat.unsqueeze(1) * w_freqs[1::2].unsqueeze(0)
    spherical_angles = torch.empty((H * W, w_half), dtype=torch.float64)
    spherical_angles[:, 0::2] = spherical_x_angles
    spherical_angles[:, 1::2] = spherical_y_angles

    w_angles = torch.where(step_mask.unsqueeze(0), linear_angles, spherical_angles)

    # L axis: all-zero positions (constant for image tokens)
    l_angles = _axis_half_angles(l_dim, torch.zeros(H * W, dtype=torch.float64))

    # Combine: [T | H | W | L] half-angles, then repeat_interleave(2) per axis
    # Flux 2 uses get_1d_rotary_pos_embed with repeat_interleave_real=True per axis,
    # then concatenates the axes.  So each axis independently does cos/sin → repeat_interleave(2).
    all_cos = []
    all_sin = []
    for angles in [t_angles, h_angles, w_angles, l_angles]:
        all_cos.append(torch.cos(angles).repeat_interleave(2, dim=-1))
        all_sin.append(torch.sin(angles).repeat_interleave(2, dim=-1))

    cos_emb = torch.cat(all_cos, dim=-1).float().unsqueeze(0)  # (1, H*W, total_dim)
    sin_emb = torch.cat(all_sin, dim=-1).float().unsqueeze(0)

    return cos_emb, sin_emb

# ---------------------------------------------------------------------------
# SFCFluxPosEmbed Wrapper
# ---------------------------------------------------------------------------


class SFCFluxPosEmbed(nn.Module):
    """Drop-in replacement for ``FluxPosEmbed`` / ``Flux2PosEmbed`` that injects SFC spherical RoPE.

    Supports two calling conventions:

    - **Flux v1**: The transformer calls ``pos_embed(cat(txt_ids, img_ids))``
      once on the full token sequence.  This wrapper splits the sequence into
      text tokens (first ``N - n_image_tokens`` rows) and image tokens (last
      ``n_image_tokens`` rows), delegates text tokens to the original module,
      and substitutes pre-computed SFC (cos, sin) tensors for the image
      portion.

    - **Flux 2**: The transformer calls ``pos_embed(img_ids)`` and
      ``pos_embed(txt_ids)`` separately.  When the input length matches
      ``n_image_tokens`` the SFC tensors are returned directly; otherwise
      the call is delegated to the original module.

    Args:
        cos_emb: Pre-computed cosine SFC RoPE tensor of shape
            ``(1, n_image_tokens, dim)``.
        sin_emb: Pre-computed sine SFC RoPE tensor of shape
            ``(1, n_image_tokens, dim)``.
        original_pos_embed: The original ``FluxPosEmbed`` or ``Flux2PosEmbed``
            module to which non-image-token IDs are delegated.
    """

    def __init__(
        self,
        cos_emb: torch.Tensor,
        sin_emb: torch.Tensor,
        original_pos_embed: nn.Module,
    ):
        super().__init__()
        self.register_buffer("cos_emb", cos_emb)
        self.register_buffer("sin_emb", sin_emb)
        self.original_pos_embed = original_pos_embed
        self._n_image_tokens = cos_emb.shape[1]

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute positional embeddings with SFC spherical RoPE for image tokens.

        Handles three calling conventions:

        - **Flux 2 (image-only)**: ``n_tokens == n_image_tokens`` — return
          SFC embeddings directly.
        - **Flux 2 (image + reference conditioning)**: ``n_tokens > n_image_tokens``
          and the first ``n_image_tokens`` IDs have ``T=0`` (generation
          latents) while the remaining IDs have ``T>0`` (reference image
          latents).  SFC RoPE is applied to the generation tokens; the
          reference tokens are delegated to the original ``pos_embed``.
        - **Flux 2 (text-only)**: ``n_tokens < n_image_tokens`` — delegate
          entirely to the original ``pos_embed``.
        - **Flux v1 (concatenated)**: ``n_tokens > n_image_tokens`` and the
          prefix IDs are text tokens — delegate text to original, apply SFC
          to image tokens.

        Args:
            ids: Token ID tensor of shape ``(N, n_axes)``.  ``N`` may equal
                ``n_image`` (Flux 2 image-only), ``n_image + n_ref``
                (Flux 2 with conditioning), ``n_text`` (Flux 2 text-only),
                or ``n_text + n_image`` (Flux v1 concatenated).

        Returns:
            Tuple ``(cos, sin)`` each of shape ``(N, dim)``.
        """
        n_tokens = ids.shape[0]

        # --- Flux 2 separate-call convention: exact match ---
        if n_tokens == self._n_image_tokens:
            # Called with generation image IDs only — return SFC embeddings
            img_cos = self.cos_emb[0, : self._n_image_tokens]
            img_sin = self.sin_emb[0, : self._n_image_tokens]
            return img_cos.to(ids.device), img_sin.to(ids.device)

        if n_tokens < self._n_image_tokens:
            # Called with text IDs only (or any non-image subset) — delegate
            return self.original_pos_embed(ids)

        # --- n_tokens > n_image_tokens ---
        # Distinguish Flux 2 (gen + ref image tokens) from Flux v1 (text + image).
        #
        # In Flux 2 with conditioning images, img_ids is:
        #   [generation_latent_ids (T=0), reference_image_ids (T=10, 20, ...)]
        # Generation tokens always have T=0 in their first coordinate.
        # Reference image tokens have T >= 10.
        #
        # In Flux v1, the concatenated sequence is [text_ids, image_ids]
        # where text_ids have T=0 but different spatial structure.
        #
        # We detect the Flux 2 conditioning case by checking whether the
        # first n_image_tokens IDs all have T=0 (generation latents) and
        # the remaining IDs have T>0 (reference images).
        gen_ids = ids[: self._n_image_tokens]
        ref_ids = ids[self._n_image_tokens :]

        is_flux2_conditioning = (
            gen_ids[:, 0].max() == 0  # generation tokens have T=0
            and ref_ids[:, 0].min() > 0  # reference tokens have T>0
        )

        if is_flux2_conditioning:
            # Flux 2 with conditioning images:
            # Apply SFC RoPE to generation tokens, delegate reference tokens
            # to the original pos_embed (they are perspective images, not ERP).
            img_cos = self.cos_emb[0, : self._n_image_tokens]
            img_sin = self.sin_emb[0, : self._n_image_tokens]

            ref_cos, ref_sin = self.original_pos_embed(ref_ids)

            return (
                torch.cat([img_cos.to(ids.device), ref_cos], dim=0),
                torch.cat([img_sin.to(ids.device), ref_sin], dim=0),
            )

        # --- Flux v1 concatenated convention ---
        n_text = n_tokens - self._n_image_tokens
        txt_ids = ids[:n_text]

        # Delegate text tokens to the original pos_embed
        txt_cos, txt_sin = self.original_pos_embed(txt_ids)

        # Use pre-computed SFC tensors for image tokens
        img_cos = self.cos_emb[0, : self._n_image_tokens]
        img_sin = self.sin_emb[0, : self._n_image_tokens]

        return (
            torch.cat([txt_cos, img_cos.to(txt_cos.device)], dim=0),
            torch.cat([txt_sin, img_sin.to(txt_sin.device)], dim=0),
        )


# ---------------------------------------------------------------------------
# Noise Wrapping — Spherical Phase Remaster
# ---------------------------------------------------------------------------


def apply_spherical_phase_remaster(x_T: torch.Tensor, radius: int = 12) -> torch.Tensor:
    """Force low-frequency phase of noise to respect spherical pole convergence.

    Standard Gaussian noise has no awareness of spherical topology. At the
    poles, where many longitudinal columns map to the same physical point,
    independent noise values create conflicting signals. This function
    replaces the low-frequency phase structure of the initial noise with a
    spherically coherent version while preserving the high-frequency texture
    and the N(0,1) variance prior.

    The algorithm:

    1. Create a "macro-structure" version of the noise by spatially pinching
       toward row means at the poles using ``alpha = 1 - cos(theta)``.
    2. Compute 2-D FFT of both the original and pinched noise.
    3. Build a circular frequency mask of the given *radius*.
    4. Replace low-frequency phase (inside the mask) with the pinched
       version's phase; keep high-frequency phase from the original.
    5. Reconstruct via inverse FFT using the **original** (untouched)
       magnitude spectrum throughout.

    All FFT operations use ``float32`` regardless of input dtype; the
    result is cast back to the original dtype before returning.

    Args:
        x_T: Initial noise tensor, shape ``(B, C, H, W)``.
        radius: Circular mask radius in frequency space. Controls how many
            low-frequency bins get the spherical phase override.  ``12`` is
            the default for 64×128 latent grids, targeting roughly the
            lowest ~450 frequency bins.

    Returns:
        Modified noise with identical magnitude spectrum but spherically
        coherent low-frequency phase. Same shape and dtype as input.
    """
    B, C, H, W = x_T.shape
    device = x_T.device
    dtype = x_T.dtype

    # Step 1: Create macro-structure by pinching poles toward row means.
    # theta maps each row to latitude in [-pi/2, pi/2] using half-pixel
    # offsets so that no row sits exactly on a pole.
    rows = torch.arange(H, dtype=torch.float32, device=device)
    theta = ((rows + 0.5) / H) * math.pi - (math.pi / 2.0)
    # alpha = 0 at equator, ~1 at poles  (sin²(theta/2) weighting)
    alpha = (1.0 - torch.cos(theta)).view(1, 1, H, 1)
    row_means = x_T.mean(dim=-1, keepdim=True)
    x_macro = x_T * (1.0 - alpha) + row_means * alpha

    # Step 2: Extract frequency-domain representations (float32 for FFT)
    F_orig = torch.fft.fftshift(torch.fft.fft2(x_T.float(), norm="ortho"), dim=(-2, -1))
    F_macro = torch.fft.fftshift(torch.fft.fft2(x_macro.float(), norm="ortho"), dim=(-2, -1))

    Mag_orig = torch.abs(F_orig)
    Phase_orig = torch.angle(F_orig)
    Phase_macro = torch.angle(F_macro)

    # Step 3: Circular low-pass mask centered at DC
    Y, X = torch.meshgrid(
        torch.arange(H, device=device) - H // 2,
        torch.arange(W, device=device) - W // 2,
        indexing="ij",
    )
    mask = (torch.sqrt(Y.float() ** 2 + X.float() ** 2) <= radius).float().view(1, 1, H, W)

    # Step 4: Inject macro phase into low frequencies only
    Phase_final = mask * Phase_macro + (1.0 - mask) * Phase_orig

    # Step 5: Reconstruct with pristine magnitude
    F_final = Mag_orig * torch.exp(1j * Phase_final)
    F_final_unshifted = torch.fft.ifftshift(F_final, dim=(-2, -1))
    x_final = torch.fft.ifft2(F_final_unshifted, norm="ortho").real

    return x_final.to(dtype)


