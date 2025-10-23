import os
import time
from typing import Dict, Any, Tuple
import torch
import numpy as np

# Reuse preview helpers similar to channel_ops

def _downscale_for_preview(img: torch.Tensor, max_dim: int = 512) -> torch.Tensor:
    if max_dim is None or max_dim <= 0:
        return img
    b, h, w, c = img.shape
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    x = img.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def _save_web_preview(img: torch.Tensor, web_dir: str, filename: str, max_dim: int = 512) -> None:
    try:
        os.makedirs(web_dir, exist_ok=True)
        try:
            from PIL import Image as _PILImage  # type: ignore
        except Exception:
            return
        img_ds = _downscale_for_preview(img.detach().to("cpu"), max_dim=max_dim)
        b, h, w, c = img_ds.shape
        if b < 1:
            return
        arr = (img_ds[0].clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        pil = _PILImage.fromarray(arr, mode="RGB")
        out_path = os.path.join(web_dir, filename)
        pil.save(out_path, format="PNG")
    except Exception as e:
        print(f"[LayerBlendingNode] Preview save failed: {e}")


def _align_batches(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 4 or b.ndim != 4:
        return a, b
    ba = a.shape[0]
    bb = b.shape[0]
    if ba == bb:
        return a, b
    m = min(ba, bb)
    return a[:m], b[:m]


def _resize_hw(x: torch.Tensor, new_hw: Tuple[int, int]) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-1] != 3:
        return x
    b, h, w, c = x.shape
    nh, nw = new_hw
    if h == nh and w == nw:
        return x
    xc = x.permute(0, 3, 1, 2)
    xr = torch.nn.functional.interpolate(xc, size=(nh, nw), mode="bilinear", align_corners=False)
    return xr.permute(0, 2, 3, 1)


def _blend_normal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return b


def _blend_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def _blend_screen(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - a) * (1.0 - b)


def _blend_overlay(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Overlay: base A, blend B
    return torch.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))


def _blend_soft_light(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Simplified soft light approximation: (1-2B)*A^2 + 2B*A
    return (1.0 - 2.0 * b) * (a * a) + 2.0 * b * a


def _blend_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def _blend_subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a - b


def _blend_darken(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def _blend_lighten(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def _blend_color_dodge(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return torch.clamp(a / torch.clamp(1.0 - b, min=eps), 0.0, 1.0)


def _blend_color_burn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return 1.0 - torch.clamp((1.0 - a) / torch.clamp(b, min=eps), 0.0, 1.0)


def _blend_linear_burn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - 1.0


def _blend_vivid_light(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # If B<0.5 -> Color Burn(A, 2B), else Color Dodge(A, 2B-1)
    low_mask = b < 0.5
    b1 = torch.clamp(2.0 * b, 0.0, 1.0)
    b2 = torch.clamp(2.0 * (b - 0.5), 0.0, 1.0)
    out_burn = _blend_color_burn(a, b1)
    out_dodge = _blend_color_dodge(a, b2)
    return torch.where(low_mask, out_burn, out_dodge)


def _blend_linear_light(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + 2.0 * b - 1.0


def _blend_pin_light(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # If B<0.5 -> min(A, 2B), else max(A, 2B-1)
    low_mask = b < 0.5
    low = torch.minimum(a, torch.clamp(2.0 * b, 0.0, 1.0))
    high = torch.maximum(a, torch.clamp(2.0 * b - 1.0, 0.0, 1.0))
    return torch.where(low_mask, low, high)


def _blend_hard_light(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(b <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))


def _blend_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(a - b)


def _blend_exclusion(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - 2.0 * a * b


def _blend_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return torch.clamp(a / torch.clamp(b, min=eps), 0.0, 1.0)


def _blend_hard_mix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Threshold of linear light approximation: A + B >= 1 -> 1 else 0
    return torch.where(a + b < 1.0, torch.zeros_like(a), torch.ones_like(a))


_BLEND_FUNCS: Dict[str, Any] = {
    "normal": _blend_normal,
    "multiply": _blend_multiply,
    "screen": _blend_screen,
    "overlay": _blend_overlay,
    "soft_light": _blend_soft_light,
    "hard_light": _blend_hard_light,
    "add": _blend_add,
    "linear_dodge": _blend_add,
    "subtract": _blend_subtract,
    "divide": _blend_divide,
    "darken": _blend_darken,
    "lighten": _blend_lighten,
    "color_dodge": _blend_color_dodge,
    "color_burn": _blend_color_burn,
    "linear_burn": _blend_linear_burn,
    "vivid_light": _blend_vivid_light,
    "linear_light": _blend_linear_light,
    "pin_light": _blend_pin_light,
    "difference": _blend_difference,
    "exclusion": _blend_exclusion,
    "hard_mix": _blend_hard_mix,
}


def _norm_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    # map synonyms and spaces to underscores
    m = m.replace("(", " ").replace(")", " ")
    m = "_".join([t for t in m.split() if t])
    synonyms = {
        "linear_dodge_add": "linear_dodge",
        "softlight": "soft_light",
        "hardlight": "hard_light",
        "colordodge": "color_dodge",
        "colorburn": "color_burn",
        "linearlight": "linear_light",
        "linearburn": "linear_burn",
        "pinlight": "pin_light",
        "hardmix": "hard_mix",
    }
    return synonyms.get(m, m)


def apply_blend(bg: torch.Tensor, fg: torch.Tensor, mode: str, opacity_255: float) -> torch.Tensor:
    # bg, fg: [B,H,W,3], values in [0,1]
    # Align batches and spatial sizes
    fg, bg = _align_batches(fg, bg)
    _, h, w, _ = bg.shape
    fg = _resize_hw(fg, (h, w))

    # Clamp inputs just in case
    bg = torch.clamp(bg, 0.0, 1.0)
    fg = torch.clamp(fg, 0.0, 1.0)

    key = _norm_mode(mode)
    blend_fn = _BLEND_FUNCS.get(key, _blend_normal)
    blended = blend_fn(bg, fg)

    # Opacity in [0,255] -> normalized
    alpha = float(opacity_255) / 255.0
    alpha = max(0.0, min(1.0, alpha))

    out = bg * (1.0 - alpha) + blended * alpha
    return torch.clamp(out, 0.0, 1.0)


class LayerBlendingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_bg": ("IMAGE",),
                "image_fg": ("IMAGE",),
                "mode": ([
                    "Normal",
                    "Multiply",
                    "Screen",
                    "Overlay",
                    "Soft Light",
                    "Hard Light",
                    "Darken",
                    "Lighten",
                    "Color Dodge",
                    "Color Burn",
                    "Linear Dodge (Add)",
                    "Linear Burn",
                    "Vivid Light",
                    "Linear Light",
                    "Pin Light",
                    "Difference",
                    "Exclusion",
                    "Add",
                    "Subtract",
                    "Divide",
                    "Hard Mix"
                ], {"default": "Normal"}),
            },
            "optional": {
                "opacity": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "preview_id": ("STRING", {"default": "A"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "preview_ref")
    FUNCTION = "run"
    CATEGORY = "image/compose"

    def run(self, image_bg, image_fg, mode, opacity=255, preview_id: str = "A"):
        out = apply_blend(image_bg, image_fg, mode, opacity)

        # Save previews for live UI: background and foreground
        this_dir = os.path.dirname(os.path.abspath(__file__))
        web_dir = os.path.join(this_dir, "web")
        safe_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in (preview_id or 'A'))
        fname_bg = f"layer_blend_bg_{safe_id}.png"
        fname_fg = f"layer_blend_fg_{safe_id}.png"
        _save_web_preview(image_bg, web_dir, filename=fname_bg, max_dim=512)
        _save_web_preview(image_fg, web_dir, filename=fname_fg, max_dim=512)

        # Send event to trigger frontend reload
        try:
            from server import PromptServer  # type: ignore
            payload = {
                "preview_id": safe_id,
                "ts": time.time(),
            }
            PromptServer.instance.send_sync("layer_blend_preview", payload)
        except Exception:
            pass

        prev_ref = _downscale_for_preview(out, max_dim=512)
        return (out, prev_ref)
