import os
import time
import math
from typing import Dict, Any, List, Tuple
import torch
import numpy as np

from .layer_blending import apply_blend, _downscale_for_preview, _save_web_preview
from .channel_ops import rgb_to_hsv, hsv_to_rgb, rgb_to_oklab, oklab_to_rgb, apply_mask_blend


EFFECT_CHOICES = [
    "Blur",
    "Halftone",
    "Color Filter",
    "Sharpen",
    "Pixelate",
    "Posterize",
    "Vignette",
    "Levels",
    "Color Balance",
    "Laplacian Sharpen",
    "Unsharp Masking",
]

LAPLACIAN_KERNELS = ["3x3 (4-neighbor)", "3x3 (8-neighbor)"]

BLUR_MODES = ["Gaussian", "Average", "Edge Average"]
LEVELS_CHANNELS = ["RGB", "Red", "Green", "Blue"]
PIXELATE_MODES = ["Pixelate", "Mosaic"]
POSTERIZE_MODES = ["RGB", "Luminance"]
POSTERIZE_DITHER_MODES = ["None", "Bayer", "Random", "Floyd-Steinberg", "Atkinson"]


def _resolve_blur_effect_name(blur_mode: str) -> str:
    """Map the new blur_mode dropdown values to the internal blur effect names
    used by _apply_blur_effect. Old workflow effect names ("Average Blur" etc.)
    pass through unchanged."""
    m = (blur_mode or "Gaussian").strip().lower()
    if m == "average":
        return "average blur"
    if m == "edge average":
        return "average edge blur"
    if m == "gaussian":
        return "gaussian blur"
    return m

SOURCE_CHOICES = [
    "Red", "Green", "Blue",
    "Red+Green", "Red+Blue", "Green+Blue",
    "Hue", "Saturation", "Value",
    "RGB", "HSV", "Oklab",
]

HALFTONE_SHAPES = [
    "Random Dots", "Cross Cut", "Dot", "Line Scalloped", "Line",
    "Line Centered", "Rhomboid", "Round", "Saddle", "Spot",
    "Spot Diamond", "Square Dot",
]

# 8x8 Bayer matrix for ordered dithering. Approximates Floyd-Steinberg-style
# error diffusion in a fully parallel (tensor-friendly) way.
_BAYER_8 = [
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
]

BLEND_MODE_CHOICES = [
    "Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light",
    "Darken", "Lighten", "Color Dodge", "Color Burn",
    "Linear Dodge (Add)", "Linear Burn", "Vivid Light", "Linear Light", "Pin Light",
    "Difference", "Exclusion", "Add", "Subtract", "Divide", "Hard Mix",
]


# Which RGB channel indices a Source label touches. HSV/Oklab handled separately.
RGB_INDEX_GROUPS: Dict[str, List[int]] = {
    "R": [0], "G": [1], "B": [2],
    "R+G": [0, 1], "R+B": [0, 2], "G+B": [1, 2],
    "RGB": [0, 1, 2],
}


# ---------- blur primitives ----------
def _box_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    r = int(radius)
    if r <= 0:
        return image
    k = 2 * r + 1
    x = image.permute(0, 3, 1, 2)
    x = torch.nn.functional.pad(x, (r, r, 0, 0), mode="replicate")
    x = torch.nn.functional.avg_pool2d(x, kernel_size=(1, k), stride=1)
    x = torch.nn.functional.pad(x, (0, 0, r, r), mode="replicate")
    x = torch.nn.functional.avg_pool2d(x, kernel_size=(k, 1), stride=1)
    return x.permute(0, 2, 3, 1)


def _bresenham_circle_offsets(r: int) -> List[Tuple[int, int]]:
    if r <= 0:
        return [(0, 0)]
    pts = set()
    x, y, d = 0, r, 1 - r
    while x <= y:
        for p in (
            (x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x),
        ):
            pts.add(p)
        x += 1
        if d < 0:
            d += 2 * x + 1
        else:
            y -= 1
            d += 2 * (x - y) + 1
    return list(pts)


def _average_edge_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    r = int(radius)
    if r <= 0:
        return image
    offsets = _bresenham_circle_offsets(r)
    n = len(offsets)
    if n == 0:
        return image
    x = image.permute(0, 3, 1, 2).contiguous()
    _, _, H, W = x.shape
    x_pad = torch.nn.functional.pad(x, (r, r, r, r), mode="replicate")
    acc = torch.zeros_like(x)
    for (dx, dy) in offsets:
        acc.add_(x_pad[..., r + dy:r + dy + H, r + dx:r + dx + W])
    acc.mul_(1.0 / float(n))
    return acc.permute(0, 2, 3, 1)


def _gaussian_blur(image: torch.Tensor, radius_param: int) -> torch.Tensor:
    rp = int(radius_param)
    if rp <= 0:
        return image
    sigma = max(0.5, float(rp) / 3.0)
    kr = max(1, int(round(3.0 * sigma)))

    x = image.permute(0, 3, 1, 2).contiguous()
    _, _, H, W = x.shape

    coords = torch.arange(-kr, kr + 1, device=x.device, dtype=x.dtype)
    k1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()
    weights = k1d.tolist()

    x_pad = torch.nn.functional.pad(x, (kr, kr, 0, 0), mode="replicate")
    acc = torch.zeros_like(x)
    for i in range(2 * kr + 1):
        acc.add_(x_pad[..., :, i:i + W], alpha=weights[i])

    x_pad = torch.nn.functional.pad(acc, (0, 0, kr, kr), mode="replicate")
    acc = torch.zeros_like(x)
    for i in range(2 * kr + 1):
        acc.add_(x_pad[..., i:i + H, :], alpha=weights[i])

    return acc.permute(0, 2, 3, 1)


def _apply_blur_effect(x: torch.Tensor, effect: str, radius: int) -> torch.Tensor:
    eff = (effect or "").strip().lower()
    if eff == "average blur":
        return _box_blur(x, radius)
    if eff == "average edge blur":
        return _average_edge_blur(x, radius)
    if eff == "gaussian blur":
        return _gaussian_blur(x, radius)
    return x


def _blur_hue(h: torch.Tensor, effect: str, radius: int) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    cs = torch.stack([torch.cos(h * two_pi), torch.sin(h * two_pi)], dim=-1)
    cs_b = _apply_blur_effect(cs, effect, radius)
    h_new = torch.atan2(cs_b[..., 1], cs_b[..., 0]) / two_pi
    return torch.remainder(h_new, 1.0)


def _blur_rgb_indices(orig: torch.Tensor, indices: List[int], effect: str, radius: int) -> torch.Tensor:
    """Blur the selected RGB channels (e.g. [0,2] for R+B), keep others original."""
    if indices == [0, 1, 2]:
        return _apply_blur_effect(orig, effect, radius)
    # Stack selected channels into a contiguous sub-tensor so the effect runs in one pass,
    # then scatter back into the cloned original.
    sub = torch.stack([orig[..., i] for i in indices], dim=-1)
    sub_b = _apply_blur_effect(sub, effect, radius)
    processed = orig.clone()
    for j, i in enumerate(indices):
        processed[..., i] = sub_b[..., j]
    return processed


# ---------- halftone ----------
def _apply_halftone(
    image: torch.Tensor,
    shape: str,
    inverse: bool,
    size: int,
    angle_deg: float,
    contrast: int,
    brightness: int,
    quality: int,
    dither: int = 100,
) -> torch.Tensor:
    device = image.device
    dtype = image.dtype
    B, H, W, _C = image.shape
    sz = max(1, int(size))

    # Brightness/contrast on the input. brightness/contrast in [-100,100].
    b_off = float(brightness) / 100.0
    c_fac = (float(contrast) + 100.0) / 100.0  # 0..2
    adj = torch.clamp((image - 0.5) * c_fac + 0.5 + b_off, 0.0, 1.0)

    # Per-pixel luminance (Rec.601). This is the "intensity to be reproduced"
    # in a classic halftone screen.
    lum = 0.299 * adj[..., 0] + 0.587 * adj[..., 1] + 0.114 * adj[..., 2]  # [B,H,W]

    angle = math.radians(float(angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    # Rotate the cell grid around the image center.
    cx_img = (float(W) - 1.0) / 2.0
    cy_img = (float(H) - 1.0) / 2.0
    xs_s = xs - cx_img
    ys_s = ys - cy_img
    rx = (xs_s * cos_a + ys_s * sin_a) / float(sz)
    ry = (-xs_s * sin_a + ys_s * cos_a) / float(sz)
    cx_cell = torch.floor(rx)
    cy_cell = torch.floor(ry)
    lx = rx - cx_cell - 0.5  # local cell coord [-0.5, 0.5]
    ly = ry - cy_cell - 0.5

    # ---- Screen function ----
    # Classic halftone: a per-pixel "screen value" in [0,1] that's high in the
    # area where the shape is densest and low elsewhere. A pixel prints black
    # where the source luminance < screen, white otherwise. As luminance
    # decreases, more pixels in the cell fall under the screen → the shape
    # grows. Different shape functions create different visual patterns.
    sh = (shape or "Dot").strip().lower()
    if sh in ("dot", "round", "spot"):
        # Round dot: screen = 1 in centre, falls off radially to 0 at the
        # cell's farthest corner (distance sqrt(0.5)).
        d = torch.sqrt(lx * lx + ly * ly) / 0.7071
        screen = 1.0 - torch.clamp(d, 0.0, 1.0)
    elif sh == "square dot":
        d = 2.0 * torch.maximum(torch.abs(lx), torch.abs(ly))
        screen = 1.0 - torch.clamp(d, 0.0, 1.0)
    elif sh in ("line", "line centered"):
        screen = 1.0 - torch.clamp(2.0 * torch.abs(ly), 0.0, 1.0)
    elif sh == "line scalloped":
        wave = torch.sin(rx * math.pi * 2.0) * 0.15
        screen = 1.0 - torch.clamp(2.0 * torch.abs(ly - wave), 0.0, 1.0)
    elif sh in ("rhomboid", "spot diamond"):
        d = 2.0 * (torch.abs(lx) + torch.abs(ly))
        screen = 1.0 - torch.clamp(d, 0.0, 1.0)
    elif sh == "cross cut":
        # Cross-shape: high along either axis through the cell centre.
        d = 2.0 * torch.minimum(torch.abs(lx), torch.abs(ly))
        screen = 1.0 - torch.clamp(d, 0.0, 1.0)
    elif sh == "saddle":
        d = 4.0 * torch.abs(lx * ly)
        screen = 1.0 - torch.clamp(d, 0.0, 1.0)
    elif sh == "random dots":
        seed = (cx_cell.to(torch.int64) * 73856093 + cy_cell.to(torch.int64) * 19349663).abs() & 0xFFFFFFFF
        dx_jit = ((seed % 1000).to(dtype) / 1000.0 - 0.5) * 0.6
        dy_jit = ((torch.div(seed, 1000, rounding_mode="floor") % 1000).to(dtype) / 1000.0 - 0.5) * 0.6
        d = torch.sqrt((lx - dx_jit) ** 2 + (ly - dy_jit) ** 2) / 0.7071
        screen = 1.0 - torch.clamp(d, 0.0, 1.0)
    else:
        d = torch.sqrt(lx * lx + ly * ly) / 0.7071
        screen = 1.0 - torch.clamp(d, 0.0, 1.0)

    # ---- Bayer-ordered dithering on the luminance ----
    # Adds a stable per-pixel offset (±d_strength*0.5) to the source luminance
    # before thresholding against the screen. With d=100 this produces the
    # noisy, error-diffusion-like tonal transitions of a real halftone print.
    d_strength = max(0, min(100, int(dither))) / 100.0
    eff_lum = lum
    if d_strength > 0.0:
        bayer = torch.tensor(_BAYER_8, device=device, dtype=dtype) / 64.0
        bayer_cell = max(1, int(sz) // 5)
        ys_i = ((torch.arange(H, device=device) // bayer_cell) % 8).long()
        xs_i = ((torch.arange(W, device=device) // bayer_cell) % 8).long()
        bayer_full = bayer[ys_i.unsqueeze(1), xs_i.unsqueeze(0)].unsqueeze(0)  # [1,H,W]
        bayer_offset = (bayer_full - 0.5) * d_strength
        eff_lum = lum + bayer_offset

    # Threshold: white where the (dithered) luminance is above the screen
    # value; black otherwise. This is the canonical halftone-screen rule.
    mask = (eff_lum > screen).to(dtype)

    # With dither at 0, fall back to a soft mask so the result doesn't look
    # like a hard threshold (no noise to smooth tonal transitions). Realistic
    # halftone always wants some dither, so the default upstream value is 100.
    if d_strength == 0.0:
        mask = torch.clamp((eff_lum - screen) * 4.0 + 0.5, 0.0, 1.0)

    if bool(inverse):
        mask = 1.0 - mask

    out = mask.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    return out


# ---------- color filter ----------
def _apply_color_filter(
    image: torch.Tensor,
    hue_deg: float,
    saturation: float,
    density: int,
    preserve_highlights: int,
) -> torch.Tensor:
    orig = torch.clamp(image, 0.0, 1.0)
    device = orig.device
    dtype = orig.dtype

    # Build the tint color from HSV (value = 1, full brightness).
    h_n = (float(hue_deg) % 360.0) / 360.0
    s_n = max(0.0, min(1.0, float(saturation)))
    hsv = torch.tensor([h_n, s_n, 1.0], device=device, dtype=dtype).view(1, 1, 1, 3)
    filter_color = hsv_to_rgb(hsv)  # [1,1,1,3]

    density_n = max(0.0, min(1.0, float(density) / 255.0))
    preserve_n = max(0.0, min(1.0, float(preserve_highlights) / 100.0))

    # Per-pixel effective density: highlights resist the tint based on preserve.
    # weight = 1 - preserve * lum  -> at preserve=1, lum=1 pixels get weight=0
    # (untouched), lum=0 pixels get weight=1 (fully affected). At preserve=0,
    # weight=1 for every pixel (uniform full density).
    lum = 0.299 * orig[..., 0:1] + 0.587 * orig[..., 1:2] + 0.114 * orig[..., 2:3]
    weight = torch.clamp(1.0 - preserve_n * lum, 0.0, 1.0)
    eff_density = density_n * weight

    tinted = orig * filter_color
    out = orig * (1.0 - eff_density) + tinted * eff_density
    return torch.clamp(out, 0.0, 1.0)


# ---------- sharpen (unsharp mask) ----------
def _apply_sharpen(image: torch.Tensor, amount: float, radius: int, threshold: int) -> torch.Tensor:
    if abs(float(amount)) < 0.001:
        return image
    r = max(1, int(radius))
    blurred = _gaussian_blur(image, r)
    diff = image - blurred
    thr = max(0, int(threshold))
    if thr > 0:
        thr_n = float(thr) / 255.0
        # Only sharpen where the per-channel detail magnitude exceeds threshold.
        keep = (diff.abs() > thr_n).to(image.dtype)
        diff = diff * keep
    out = image + diff * (float(amount) / 100.0)
    return torch.clamp(out, 0.0, 1.0)


# ---------- laplacian sharpen ----------
def _laplacian(image: torch.Tensor, kernel: str) -> torch.Tensor:
    """Second-derivative (Laplacian) of each RGB channel. Uses a 4-neighbor
    or 8-neighbor 3x3 kernel with replicate padding so edges stay stable."""
    x = image.permute(0, 3, 1, 2).contiguous()  # B,C,H,W
    C = x.shape[1]
    if (kernel or "").find("4") >= 0:
        k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                         device=x.device, dtype=x.dtype)
    else:
        k = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]],
                         device=x.device, dtype=x.dtype)
    k = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    lap = torch.nn.functional.conv2d(x_pad, k, groups=C)
    return lap.permute(0, 2, 3, 1)


def _apply_laplacian_sharpen(image: torch.Tensor, amount: float, kernel: str) -> torch.Tensor:
    if abs(float(amount)) < 0.001:
        return image
    lap = _laplacian(image, kernel)
    # Sharpen rule g = f - k*(Laplacian f): the kernel's negative centre makes
    # subtraction lift peaks and deepen troughs, boosting local contrast.
    out = image - float(amount) * lap
    return torch.clamp(out, 0.0, 1.0)


# ---------- unsharp masking (Gaussian) ----------
def _apply_unsharp_mask(image: torch.Tensor, amount: float, radius: int, threshold: int) -> torch.Tensor:
    """Classic unsharp mask (https://en.wikipedia.org/wiki/Unsharp_masking):
    sharpened = original + amount * (original - blurred), gated by a per-channel
    threshold. `amount` is a direct multiplier (1.0 = 100%)."""
    if abs(float(amount)) < 0.001:
        return image
    r = max(1, int(radius))
    blurred = _gaussian_blur(image, r)
    diff = image - blurred
    thr = max(0, int(threshold))
    if thr > 0:
        thr_n = float(thr) / 255.0
        keep = (diff.abs() > thr_n).to(image.dtype)
        diff = diff * keep
    out = image + diff * float(amount)
    return torch.clamp(out, 0.0, 1.0)


# ---------- pixelate / mosaic ----------
def _apply_pixelate(image: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    bsize = max(1, int(size))
    B, H, W, C = image.shape
    nh = max(1, H // bsize)
    nw = max(1, W // bsize)
    x = image.permute(0, 3, 1, 2)
    x_small = torch.nn.functional.adaptive_avg_pool2d(x, (nh, nw))
    x_big = torch.nn.functional.interpolate(x_small, size=(H, W), mode="nearest")
    out = x_big.permute(0, 2, 3, 1)

    if (mode or "").strip().lower() == "mosaic" and bsize >= 3:
        device = image.device
        dtype = image.dtype
        ys = torch.arange(H, device=device) % bsize
        xs = torch.arange(W, device=device) % bsize
        grout_y = (ys < 1).to(dtype)
        grout_x = (xs < 1).to(dtype)
        mask = torch.maximum(grout_y.unsqueeze(1), grout_x.unsqueeze(0))  # [H,W]
        mask = mask.view(1, H, W, 1)
        out = out * (1.0 - mask * 0.7)
    return out


# ---------- posterize ----------
def _posterize_quantise(
    x: torch.Tensor, step: float, dither_mode: str, d_strength: float,
) -> torch.Tensor:
    """Round `x` to `step+1` equally-spaced levels in [0,1] using the chosen
    dithering strategy. Operates on any [B,H,W,C] tensor."""
    device = x.device
    dtype = x.dtype
    dm = (dither_mode or "None").strip().lower()

    if dm in ("none", "") or d_strength <= 0.0 or dm not in ("bayer", "random", "floyd-steinberg", "atkinson"):
        return torch.round(torch.clamp(x, 0.0, 1.0) * step) / step

    if dm == "bayer":
        _, H, W, _ = x.shape
        bayer = torch.tensor(_BAYER_8, device=device, dtype=dtype) / 64.0
        ys_i = (torch.arange(H, device=device) % 8).long()
        xs_i = (torch.arange(W, device=device) % 8).long()
        bayer_full = bayer[ys_i.unsqueeze(1), xs_i.unsqueeze(0)]
        bayer_full = bayer_full.unsqueeze(0).unsqueeze(-1)
        offset = (bayer_full - 0.5) * (d_strength / max(1.0, step))
        return torch.round(torch.clamp(x + offset, 0.0, 1.0) * step) / step

    if dm == "random":
        noise = (torch.rand_like(x) - 0.5) * (d_strength / max(1.0, step))
        return torch.round(torch.clamp(x + noise, 0.0, 1.0) * step) / step

    # Error-diffusion variants (Floyd-Steinberg, Atkinson) — fundamentally
    # sequential. Run on CPU via numpy for speed; strength is ignored (these
    # algorithms always propagate full error).
    arr = x.detach().cpu().numpy().astype(np.float32).copy()
    B_, H_, W_, C_ = arr.shape
    if dm == "floyd-steinberg":
        for b in range(B_):
            for c in range(C_):
                plane = arr[b, :, :, c]
                for y in range(H_):
                    for xx in range(W_):
                        old = plane[y, xx]
                        new = round(old * step) / step
                        err = old - new
                        plane[y, xx] = new
                        if xx + 1 < W_:
                            plane[y, xx + 1] += err * (7.0 / 16.0)
                        if y + 1 < H_:
                            if xx > 0:
                                plane[y + 1, xx - 1] += err * (3.0 / 16.0)
                            plane[y + 1, xx] += err * (5.0 / 16.0)
                            if xx + 1 < W_:
                                plane[y + 1, xx + 1] += err * (1.0 / 16.0)
    else:  # atkinson
        for b in range(B_):
            for c in range(C_):
                plane = arr[b, :, :, c]
                for y in range(H_):
                    for xx in range(W_):
                        old = plane[y, xx]
                        new = round(old * step) / step
                        err = (old - new) / 8.0
                        plane[y, xx] = new
                        if xx + 1 < W_: plane[y, xx + 1] += err
                        if xx + 2 < W_: plane[y, xx + 2] += err
                        if y + 1 < H_:
                            if xx > 0: plane[y + 1, xx - 1] += err
                            plane[y + 1, xx] += err
                            if xx + 1 < W_: plane[y + 1, xx + 1] += err
                        if y + 2 < H_:
                            plane[y + 2, xx] += err
    np.clip(arr, 0.0, 1.0, out=arr)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _apply_posterize(
    image: torch.Tensor,
    levels: int,
    mode: str = "RGB",
    dither: int = 0,
    dither_mode: str = "None",
) -> torch.Tensor:
    l = max(2, min(64, int(levels)))
    step = float(l - 1)
    m = (mode or "RGB").strip().lower()
    d_strength = max(0, min(100, int(dither))) / 100.0

    if m == "luminance":
        lum = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]
        lum_q = _posterize_quantise(lum.expand(-1, -1, -1, 1), step, dither_mode, d_strength)
        ratio = torch.where(lum > 1e-4, lum_q / (lum + 1e-8), torch.ones_like(lum))
        return torch.clamp(image * ratio, 0.0, 1.0)

    return torch.clamp(_posterize_quantise(image, step, dither_mode, d_strength), 0.0, 1.0)


# ---------- vignette ----------
def _apply_vignette(
    image: torch.Tensor,
    amount: float, size: float, feather: float, roundness: float,
    center_x: float = 50.0, center_y: float = 50.0,
) -> torch.Tensor:
    B, H, W, C = image.shape
    device = image.device
    dtype = image.dtype

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    cx_px = (float(center_x) / 100.0) * float(max(1, W - 1))
    cy_px = (float(center_y) / 100.0) * float(max(1, H - 1))
    nx = (xs - cx_px) / max(1.0, (W - 1) / 2.0)
    ny = (ys - cy_px) / max(1.0, (H - 1) / 2.0)

    r_n = float(roundness) / 100.0
    if r_n > 0:
        # Stretch in Y: makes vignette taller (oval portrait)
        ny = ny * (1.0 + r_n)
    elif r_n < 0:
        nx = nx * (1.0 - r_n)

    dist = torch.sqrt(nx * nx + ny * ny)

    inner = max(0.0, float(size) / 100.0)
    feather_n = max(0.001, float(feather) / 100.0)
    outer = inner + feather_n

    factor = torch.clamp((outer - dist) / (outer - inner + 1e-6), 0.0, 1.0)
    factor = factor * factor * (3.0 - 2.0 * factor)  # smoothstep
    factor = factor.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]

    amt = float(amount) / 100.0
    if amt >= 0:
        out = image * (1.0 - (1.0 - factor) * amt)
    else:
        out = image + (1.0 - image) * (1.0 - factor) * (-amt)
    return torch.clamp(out, 0.0, 1.0)


# ---------- levels ----------
def _apply_levels(
    image: torch.Tensor,
    channel: str,
    in_black: int,
    in_white: int,
    gamma: float,
    out_black: int,
    out_white: int,
) -> torch.Tensor:
    in_b = float(in_black) / 255.0
    in_w = float(in_white) / 255.0
    out_b = float(out_black) / 255.0
    out_w = float(out_white) / 255.0
    g = max(0.01, float(gamma))
    in_range = max(1.0 / 255.0, in_w - in_b)

    def apply_to(c: torch.Tensor) -> torch.Tensor:
        x = (c - in_b) / in_range
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.pow(x, 1.0 / g)
        x = out_b + x * (out_w - out_b)
        return torch.clamp(x, 0.0, 1.0)

    ch = (channel or "RGB").strip().upper()
    if ch == "RGB":
        return apply_to(image)
    idx_map = {"R": 0, "RED": 0, "G": 1, "GREEN": 1, "B": 2, "BLUE": 2}
    if ch in idx_map:
        idx = idx_map[ch]
        out = image.clone()
        out[..., idx] = apply_to(image[..., idx])
        return out
    return apply_to(image)


# ---------- color balance ----------
def _apply_color_balance(
    image: torch.Tensor,
    sh_r: float, sh_g: float, sh_b: float,
    mid_r: float, mid_g: float, mid_b: float,
    hi_r: float, hi_g: float, hi_b: float,
) -> torch.Tensor:
    lum = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]

    shadow_w = torch.clamp(1.0 - 2.0 * lum, 0.0, 1.0)
    midtone_w = 1.0 - torch.abs(2.0 * lum - 1.0)
    highlight_w = torch.clamp(2.0 * lum - 1.0, 0.0, 1.0)

    s = 1.0 / 200.0  # so a full +100 nudge moves a channel by at most 0.5
    shift_r = (float(sh_r) * s) * shadow_w + (float(mid_r) * s) * midtone_w + (float(hi_r) * s) * highlight_w
    shift_g = (float(sh_g) * s) * shadow_w + (float(mid_g) * s) * midtone_w + (float(hi_g) * s) * highlight_w
    shift_b = (float(sh_b) * s) * shadow_w + (float(mid_b) * s) * midtone_w + (float(hi_b) * s) * highlight_w

    r = image[..., 0:1] + shift_r
    g = image[..., 1:2] + shift_g
    b = image[..., 2:3] + shift_b
    out = torch.cat([r, g, b], dim=-1)
    return torch.clamp(out, 0.0, 1.0)


# ---------- main dispatch ----------
def apply_image_effect(
    image: torch.Tensor,
    effect: str,
    source: str,
    radius: int,
    blend_mode: str,
    halftone_shape: str = "Dot",
    halftone_inverse: bool = False,
    halftone_size: int = 18,
    halftone_angle: float = 15.0,
    halftone_contrast: int = 0,
    halftone_brightness: int = 0,
    halftone_quality: int = 2,
    halftone_dither: int = 100,
    color_hue: int = 0,
    color_saturation: float = 1.0,
    color_density: int = 128,
    color_preserve_highlights: int = 50,
    sharpen_amount: int = 50,
    sharpen_radius: int = 2,
    sharpen_threshold: int = 0,
    pixelate_size: int = 16,
    pixelate_mode: str = "Pixelate",
    posterize_levels: int = 4,
    vignette_amount: int = 50,
    vignette_size: int = 50,
    vignette_feather: int = 30,
    vignette_roundness: int = 0,
    levels_channel: str = "RGB",
    levels_in_black: int = 0,
    levels_in_white: int = 255,
    levels_gamma: float = 1.0,
    levels_out_black: int = 0,
    levels_out_white: int = 255,
    cb_shadow_red: int = 0, cb_shadow_green: int = 0, cb_shadow_blue: int = 0,
    cb_midtone_red: int = 0, cb_midtone_green: int = 0, cb_midtone_blue: int = 0,
    cb_highlight_red: int = 0, cb_highlight_green: int = 0, cb_highlight_blue: int = 0,
    blur_mode: str = "Gaussian",
    posterize_mode: str = "RGB",
    posterize_dither: int = 0,
    posterize_dither_mode: str = "None",
    vignette_center_x: int = 50,
    vignette_center_y: int = 50,
    laplacian_amount: float = 1.0,
    laplacian_kernel: str = "3x3 (8-neighbor)",
    usm_amount: float = 1.0,
    usm_radius: int = 3,
    usm_threshold: int = 0,
    mask=None,
) -> torch.Tensor:
    orig = torch.clamp(image, 0.0, 1.0)
    eff = (effect or "").strip().lower()

    if eff == "halftone":
        # Dithering is always on (Bayer 8x8 ordered dithering, scaled by size).
        # The halftone_dither widget is kept in INPUT_TYPES for back-compat
        # but its value is ignored.
        processed = _apply_halftone(
            orig, halftone_shape, halftone_inverse,
            halftone_size, halftone_angle,
            halftone_contrast, halftone_brightness, halftone_quality,
            100,
        )
    elif eff == "color filter":
        processed = _apply_color_filter(
            orig, color_hue, color_saturation, color_density, color_preserve_highlights,
        )
    elif eff == "sharpen":
        processed = _apply_sharpen(orig, sharpen_amount, sharpen_radius, sharpen_threshold)
    elif eff == "laplacian sharpen":
        processed = _apply_laplacian_sharpen(orig, laplacian_amount, laplacian_kernel)
    elif eff == "unsharp masking":
        processed = _apply_unsharp_mask(orig, usm_amount, usm_radius, usm_threshold)
    elif eff == "pixelate":
        processed = _apply_pixelate(orig, pixelate_size, pixelate_mode)
    elif eff == "posterize":
        processed = _apply_posterize(orig, posterize_levels, posterize_mode, posterize_dither, posterize_dither_mode)
    elif eff == "vignette":
        processed = _apply_vignette(
            orig, vignette_amount, vignette_size, vignette_feather, vignette_roundness,
            vignette_center_x, vignette_center_y,
        )
    elif eff == "levels":
        processed = _apply_levels(
            orig, levels_channel,
            levels_in_black, levels_in_white, levels_gamma,
            levels_out_black, levels_out_white,
        )
    elif eff == "color balance":
        processed = _apply_color_balance(
            orig,
            cb_shadow_red, cb_shadow_green, cb_shadow_blue,
            cb_midtone_red, cb_midtone_green, cb_midtone_blue,
            cb_highlight_red, cb_highlight_green, cb_highlight_blue,
        )
    else:
        # Resolve the actual blur algorithm. "Blur" effect uses blur_mode;
        # legacy effect names ("Average Blur" etc.) from old workflows are
        # accepted as direct aliases.
        if eff == "blur":
            blur_effect = _resolve_blur_effect_name(blur_mode)
        else:
            blur_effect = eff  # legacy: "average blur" / "gaussian blur" / "average edge blur"

        name_map = {
            "RED": "R", "GREEN": "G", "BLUE": "B",
            "RED+GREEN": "R+G", "RED+BLUE": "R+B", "GREEN+BLUE": "G+B",
            "HUE": "H", "SATURATION": "S", "VALUE": "V",
            "RGB": "RGB", "HSV": "HSV", "OKLAB": "OKLAB",
        }
        s = name_map.get((source or "RGB").strip().upper(), "RGB")

        if s in RGB_INDEX_GROUPS:
            processed = _blur_rgb_indices(orig, RGB_INDEX_GROUPS[s], blur_effect, radius)
        elif s in ("H", "S", "V"):
            hsv = rgb_to_hsv(orig).clone()
            idx = {"H": 0, "S": 1, "V": 2}[s]
            if idx == 0:
                hsv[..., 0] = _blur_hue(hsv[..., 0], blur_effect, radius)
            else:
                ch = hsv[..., idx:idx + 1]
                hsv[..., idx:idx + 1] = torch.clamp(_apply_blur_effect(ch, blur_effect, radius), 0.0, 1.0)
            processed = hsv_to_rgb(hsv)
        elif s == "HSV":
            hsv = rgb_to_hsv(orig)
            h_b = _blur_hue(hsv[..., 0], blur_effect, radius)
            sv_b = torch.clamp(_apply_blur_effect(hsv[..., 1:3], blur_effect, radius), 0.0, 1.0)
            hsv2 = torch.stack([h_b, sv_b[..., 0], sv_b[..., 1]], dim=-1)
            processed = hsv_to_rgb(hsv2)
        elif s == "OKLAB":
            lab = rgb_to_oklab(orig)
            lab_b = _apply_blur_effect(lab, blur_effect, radius)
            processed = oklab_to_rgb(lab_b)
        else:
            processed = _apply_blur_effect(orig, blur_effect, radius)

    out = apply_blend(orig, processed, blend_mode, 255)
    # If a mask was provided, restrict the entire effect (including the blend)
    # to the masked region — pixels outside the mask keep the original image.
    if mask is not None:
        out = apply_mask_blend(orig, out, mask)
    return torch.clamp(out, 0.0, 1.0)


class ImageEffectsNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Widget order is append-only across versions so older workflows keep
        # mapping their saved positional widgets_values to the right widgets.
        return {
            "required": {
                "image": ("IMAGE",),
                "effect": (EFFECT_CHOICES, {"default": "Blur"}),
                "radius": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1}),
                "source": (SOURCE_CHOICES, {"default": "RGB"}),
                "blend_mode": (BLEND_MODE_CHOICES, {"default": "Normal"}),
            },
            "optional": {
                # Halftone-specific widgets. Hidden by JS unless effect == Halftone.
                "halftone_shape": (HALFTONE_SHAPES, {"default": "Dot"}),
                "halftone_inverse": ("BOOLEAN", {"default": False}),
                "halftone_size": ("INT", {"default": 12, "min": 1, "max": 100, "step": 1}),
                "halftone_angle": ("FLOAT", {"default": 15.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "halftone_contrast": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "halftone_brightness": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "halftone_quality": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "halftone_dither": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                # Color Filter widgets. color_saturation is kept hidden in the
                # UI and only written by the custom color-wheel widget.
                "color_hue": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
                "color_density": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "color_preserve_highlights": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "color_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                # Sharpen (unsharp mask) widgets. Negative amount = soften.
                "sharpen_amount": ("INT", {"default": 50, "min": -200, "max": 200, "step": 1}),
                "sharpen_radius": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "sharpen_threshold": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                # Pixelate / Mosaic.
                "pixelate_size": ("INT", {"default": 16, "min": 1, "max": 200, "step": 1}),
                "pixelate_mode": (PIXELATE_MODES, {"default": "Pixelate"}),
                # Posterize.
                "posterize_levels": ("INT", {"default": 4, "min": 2, "max": 32, "step": 1}),
                # Vignette.
                "vignette_amount": ("INT", {"default": 50, "min": -100, "max": 100, "step": 1}),
                "vignette_size": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "vignette_feather": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "vignette_roundness": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                # Levels.
                "levels_channel": (LEVELS_CHANNELS, {"default": "RGB"}),
                "levels_in_black": ("INT", {"default": 0, "min": 0, "max": 254, "step": 1}),
                "levels_in_white": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),
                "levels_gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 9.99, "step": 0.01}),
                "levels_out_black": ("INT", {"default": 0, "min": 0, "max": 254, "step": 1}),
                "levels_out_white": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),
                # Color Balance (per tonal range, per channel; negative = complementary).
                "cb_shadow_red":      ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_shadow_green":    ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_shadow_blue":     ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_midtone_red":     ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_midtone_green":   ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_midtone_blue":    ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_highlight_red":   ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_highlight_green": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "cb_highlight_blue":  ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                # Blur mode (for the combined "Blur" effect).
                "blur_mode": (BLUR_MODES, {"default": "Gaussian"}),
                # Posterize extras.
                "posterize_mode": (POSTERIZE_MODES, {"default": "RGB"}),
                "posterize_dither": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                # Vignette center (0-100 percent of image dimensions).
                "vignette_center_x": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "vignette_center_y": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "posterize_dither_mode": (POSTERIZE_DITHER_MODES, {"default": "None"}),
                # Laplacian Sharpen.
                "laplacian_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "laplacian_kernel": (LAPLACIAN_KERNELS, {"default": "3x3 (8-neighbor)"}),
                # Unsharp Masking (Gaussian). amount is a direct multiplier.
                "usm_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "usm_radius": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "usm_threshold": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "mask": ("MASK",),
                "preview_id": ("STRING", {"default": "A"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/processing"

    def run(self, image, effect, radius, source, blend_mode,
            preview_id: str = "A", **kwargs):
        # **kwargs lets us add new effect-specific widgets in INPUT_TYPES["optional"]
        # without changing this signature.
        out = apply_image_effect(
            image, effect, source, radius, blend_mode,
            halftone_shape=kwargs.get("halftone_shape", "Dot"),
            halftone_inverse=bool(kwargs.get("halftone_inverse", False)),
            halftone_size=int(kwargs.get("halftone_size", 18)),
            halftone_angle=float(kwargs.get("halftone_angle", 15.0)),
            halftone_contrast=int(kwargs.get("halftone_contrast", 0)),
            halftone_brightness=int(kwargs.get("halftone_brightness", 0)),
            halftone_quality=int(kwargs.get("halftone_quality", 2)),
            halftone_dither=int(kwargs.get("halftone_dither", 100)),
            color_hue=int(kwargs.get("color_hue", 0)),
            # Saturation is fixed at 1.0 — the color wheel only writes hue +
            # density now. The widget remains in INPUT_TYPES for positional
            # widgets_values back-compat, but its value is ignored.
            color_saturation=1.0,
            color_density=int(kwargs.get("color_density", 128)),
            color_preserve_highlights=int(kwargs.get("color_preserve_highlights", 50)),
            sharpen_amount=int(kwargs.get("sharpen_amount", 50)),
            sharpen_radius=int(kwargs.get("sharpen_radius", 2)),
            sharpen_threshold=int(kwargs.get("sharpen_threshold", 0)),
            pixelate_size=int(kwargs.get("pixelate_size", 16)),
            pixelate_mode=str(kwargs.get("pixelate_mode", "Pixelate")),
            posterize_levels=int(kwargs.get("posterize_levels", 4)),
            vignette_amount=int(kwargs.get("vignette_amount", 50)),
            vignette_size=int(kwargs.get("vignette_size", 50)),
            vignette_feather=int(kwargs.get("vignette_feather", 30)),
            vignette_roundness=int(kwargs.get("vignette_roundness", 0)),
            levels_channel=str(kwargs.get("levels_channel", "RGB")),
            levels_in_black=int(kwargs.get("levels_in_black", 0)),
            levels_in_white=int(kwargs.get("levels_in_white", 255)),
            levels_gamma=float(kwargs.get("levels_gamma", 1.0)),
            levels_out_black=int(kwargs.get("levels_out_black", 0)),
            levels_out_white=int(kwargs.get("levels_out_white", 255)),
            cb_shadow_red=int(kwargs.get("cb_shadow_red", 0)),
            cb_shadow_green=int(kwargs.get("cb_shadow_green", 0)),
            cb_shadow_blue=int(kwargs.get("cb_shadow_blue", 0)),
            cb_midtone_red=int(kwargs.get("cb_midtone_red", 0)),
            cb_midtone_green=int(kwargs.get("cb_midtone_green", 0)),
            cb_midtone_blue=int(kwargs.get("cb_midtone_blue", 0)),
            cb_highlight_red=int(kwargs.get("cb_highlight_red", 0)),
            cb_highlight_green=int(kwargs.get("cb_highlight_green", 0)),
            cb_highlight_blue=int(kwargs.get("cb_highlight_blue", 0)),
            blur_mode=str(kwargs.get("blur_mode", "Gaussian")),
            posterize_mode=str(kwargs.get("posterize_mode", "RGB")),
            posterize_dither=int(kwargs.get("posterize_dither", 0)),
            vignette_center_x=int(kwargs.get("vignette_center_x", 50)),
            vignette_center_y=int(kwargs.get("vignette_center_y", 50)),
            posterize_dither_mode=str(kwargs.get("posterize_dither_mode", "None")),
            laplacian_amount=float(kwargs.get("laplacian_amount", 1.0)),
            laplacian_kernel=str(kwargs.get("laplacian_kernel", "3x3 (8-neighbor)")),
            usm_amount=float(kwargs.get("usm_amount", 1.0)),
            usm_radius=int(kwargs.get("usm_radius", 3)),
            usm_threshold=int(kwargs.get("usm_threshold", 0)),
            mask=kwargs.get("mask"),
        )

        this_dir = os.path.dirname(os.path.abspath(__file__))
        web_dir = os.path.join(this_dir, "web")
        safe_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in (preview_id or 'A'))
        fname_src = f"image_effects_src_{safe_id}.png"
        _save_web_preview(image, web_dir, filename=fname_src, max_dim=512)

        try:
            from server import PromptServer  # type: ignore
            try:
                orig_h = int(image.shape[1])
                orig_w = int(image.shape[2])
            except Exception:
                orig_h = orig_w = 0
            payload = {
                "preview_id": safe_id,
                "ts": time.time(),
                "orig_w": orig_w,
                "orig_h": orig_h,
            }
            PromptServer.instance.send_sync("image_effects_preview", payload)
        except Exception:
            pass

        return (out,)
