import os
import time
import torch
import numpy as np

from .channel_ops import rgb_to_oklab, oklab_to_rgb
from .layer_blending import _save_web_preview


# Color matching methods. "LAB" transfers mean/std in the perceptual Oklab
# space (a Lab-style space already used elsewhere in this pack); "RGB" does the
# same transfer directly on RGB; "Histogram" matches each channel's cumulative
# distribution to the reference.
COLOR_MATCH_METHODS = ["LAB", "RGB", "Histogram"]


def _match_mean_std(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reinhard-style statistics transfer: rescale each channel of `src` so its
    per-channel mean/std match the reference. Stats are taken over all pixels
    (batch + spatial). Population std (unbiased=False) is used so the JS preview
    can reproduce the exact same numbers."""
    dims = (0, 1, 2)
    src_mean = src.mean(dim=dims, keepdim=True)
    src_std = src.std(dim=dims, keepdim=True, unbiased=False)
    ref_mean = ref.mean(dim=dims, keepdim=True)
    ref_std = ref.std(dim=dims, keepdim=True, unbiased=False)
    return (src - src_mean) * (ref_std / (src_std + 1e-6)) + ref_mean


def color_match_lab(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    src_lab = rgb_to_oklab(image)
    ref_lab = rgb_to_oklab(reference)
    out_lab = _match_mean_std(src_lab, ref_lab)
    return oklab_to_rgb(out_lab)


def color_match_rgb(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.clamp(_match_mean_std(image, reference), 0.0, 1.0)


def _hist_match_channel(src_q: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    """Match `src_q` (uint8 levels) to the reference's cumulative distribution.
    Returns a mapped plane in [0,1]. Uses searchsorted so it matches the JS
    preview's LUT construction exactly."""
    s_hist = np.bincount(src_q, minlength=256).astype(np.float64)
    r_hist = np.bincount(ref_q, minlength=256).astype(np.float64)
    s_cdf = np.cumsum(s_hist)
    r_cdf = np.cumsum(r_hist)
    s_cdf /= (s_cdf[-1] + 1e-12)
    r_cdf /= (r_cdf[-1] + 1e-12)
    # For each source CDF value, first reference level whose CDF is >= it.
    lut = np.searchsorted(r_cdf, s_cdf, side="left")
    lut = np.clip(lut, 0, 255).astype(np.float64) / 255.0
    return lut[src_q]


def color_match_histogram(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dev, dt = image.device, image.dtype
    img = image.detach().cpu().numpy()
    ref = reference.detach().cpu().numpy()
    B, H, W, C = img.shape
    out = np.empty_like(img)
    for c in range(C):
        src_q = np.clip((img[..., c].reshape(-1) * 255.0), 0, 255).astype(np.int64)
        ref_q = np.clip((ref[..., c].reshape(-1) * 255.0), 0, 255).astype(np.int64)
        out[..., c] = _hist_match_channel(src_q, ref_q).reshape(B, H, W)
    return torch.from_numpy(out).to(device=dev, dtype=dt).clamp(0.0, 1.0)


def apply_color_match(image: torch.Tensor, reference: torch.Tensor, method: str = "LAB") -> torch.Tensor:
    image = torch.clamp(image, 0.0, 1.0)
    reference = torch.clamp(reference, 0.0, 1.0)
    m = (method or "LAB").strip().lower()
    if m == "rgb":
        return color_match_rgb(image, reference)
    if m == "histogram":
        return color_match_histogram(image, reference)
    return color_match_lab(image, reference)


class ColorMatchingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": (COLOR_MATCH_METHODS, {"default": "LAB"}),
            },
            "optional": {
                "preview_id": ("STRING", {"default": "A"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/processing"

    def run(self, image, reference, method="LAB", preview_id: str = "A"):
        out = apply_color_match(image, reference, method)

        # Save both inputs so the frontend can recompute the match live when the
        # method changes (mirrors the Layer Blending preview pattern).
        this_dir = os.path.dirname(os.path.abspath(__file__))
        web_dir = os.path.join(this_dir, "web")
        safe_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in (preview_id or 'A'))
        _save_web_preview(image, web_dir, filename=f"color_match_src_{safe_id}.png", max_dim=512)
        _save_web_preview(reference, web_dir, filename=f"color_match_ref_{safe_id}.png", max_dim=512)

        try:
            from server import PromptServer  # type: ignore
            payload = {
                "preview_id": safe_id,
                "ts": time.time(),
            }
            PromptServer.instance.send_sync("color_match_preview", payload)
        except Exception:
            pass

        return (out,)
