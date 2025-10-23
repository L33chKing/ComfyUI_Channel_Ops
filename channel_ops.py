import os
import time
from typing import Optional, Tuple
import torch
import numpy as np


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    # rgb in [0,1], shape [B,H,W,3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc, _ = torch.max(rgb, dim=-1)
    minc, _ = torch.min(rgb, dim=-1)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / (maxc + 1e-8), torch.zeros_like(maxc))

    # Hue computation
    h = torch.zeros_like(maxc)
    mask = delta > 1e-8
    # Avoid division by zero by adding epsilon
    rc = ((maxc - r) / (delta + 1e-8)).masked_fill(~mask, 0)
    gc = ((maxc - g) / (delta + 1e-8)).masked_fill(~mask, 0)
    bc = ((maxc - b) / (delta + 1e-8)).masked_fill(~mask, 0)

    cond_r = (maxc == r) & mask
    cond_g = (maxc == g) & mask
    cond_b = (maxc == b) & mask
    h = torch.where(cond_r, (bc - gc) / 6.0 % 1.0, h)
    h = torch.where(cond_g, (2.0 + rc - bc) / 6.0 % 1.0, h)
    h = torch.where(cond_b, (4.0 + gc - rc) / 6.0 % 1.0, h)

    hsv = torch.stack([h, s, v], dim=-1)
    return hsv


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    # hsv: h in [0,1], s in [0,1], v in [0,1], shape [B,H,W,3]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - i.to(h6.dtype)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = torch.remainder(i, 6)
    r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
    g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))

    rgb = torch.stack([r, g, b], dim=-1)
    return torch.clamp(rgb, 0.0, 1.0)


# --- Minimal Oklab conversions (stand-alone) ---
def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: [B,H,W,3] in sRGB [0,1]
    b, h, w, c = rgb.shape
    rgb_flat = rgb.reshape(-1, 3)
    # sRGB to linear
    mask = rgb_flat <= 0.04045
    rgb_lin = torch.where(mask, rgb_flat / 12.92, ((rgb_flat + 0.055) / 1.055) ** 2.4)

    M1 = torch.tensor([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ], device=rgb.device, dtype=rgb.dtype)
    lms = torch.matmul(rgb_lin, M1.T)
    lms_cbrt = torch.sign(lms) * torch.abs(lms) ** (1.0 / 3.0)

    M2 = torch.tensor([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ], device=rgb.device, dtype=rgb.dtype)
    lab = torch.matmul(lms_cbrt, M2.T)
    return lab.reshape(b, h, w, 3)


def oklab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    b, h, w, c = lab.shape
    lab_flat = lab.reshape(-1, 3)

    M2_inv = torch.tensor([
        [1.0000000000, 0.3963377774, 0.2158037573],
        [1.0000000000, -0.1055613458, -0.0638541728],
        [1.0000000000, -0.0894841775, -1.2914855480],
    ], device=lab.device, dtype=lab.dtype)
    lms_cbrt = torch.matmul(lab_flat, M2_inv.T)
    lms = torch.sign(lms_cbrt) * torch.abs(lms_cbrt) ** 3

    M1_inv = torch.tensor([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ], device=lab.device, dtype=lab.dtype)
    rgb_lin = torch.matmul(lms, M1_inv.T)

    # linear to sRGB
    mask = rgb_lin <= 0.0031308
    rgb = torch.where(mask, rgb_lin * 12.92, 1.055 * torch.clamp(rgb_lin, min=0.0) ** (1.0 / 2.4) - 0.055)
    rgb = rgb.reshape(b, h, w, 3)
    return torch.clamp(rgb, 0.0, 1.0)


def _apply_op(x: torch.Tensor, op: str, amount: float) -> torch.Tensor:
    if op == "set":
        return torch.full_like(x, amount)
    if op == "add":
        return x + amount
    if op == "subtract":
        return x - amount
    if op == "multiply":
        # amount is raw factor in [0,255]; 2 means 2x, 255 means 255x
        factor = amount
        return x * factor
    if op == "divide":
        # amount is raw factor in [0,255]; 2 means divide by 2, 255 means divide by 255
        factor = amount
        return x / (factor + 1e-8)
    if op == "clamp_max":
        return torch.minimum(x, torch.tensor(amount, device=x.device, dtype=x.dtype))
    if op == "clamp_min":
        return torch.maximum(x, torch.tensor(amount, device=x.device, dtype=x.dtype))
    if op == "truncate":
        # amount is step size in normalized units [0,1]; 0 -> no change, >0 -> round to multiples of step
        step = float(amount)
        if step <= 0.0:
            return x
        return torch.round(x / step) * step
    if op == "contrast":
        # amount is normalized [0,1]; piecewise scaling around 0.5
        a = float(amount)
        denom = max(1e-6, 1.0 - a)
        k = 1.0 / denom
        below = (x < 0.5)
        x_scaled = torch.where(below, x * (2.0 - k), torch.where(x > 0.5, x * k, x))
        return torch.clamp(x_scaled, 0.0, 1.0)
    return x


def _resize_hw(x: torch.Tensor, new_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize image tensor [B,H,W,C] to new (H,W) with bilinear sampling."""
    if x.ndim != 4 or x.shape[-1] != 3:
        return x
    b, h, w, c = x.shape
    nh, nw = new_hw
    if h == nh and w == nw:
        return x
    xc = x.permute(0, 3, 1, 2)  # B,C,H,W
    xr = torch.nn.functional.interpolate(xc, size=(nh, nw), mode="bilinear", align_corners=False)
    return xr.permute(0, 2, 3, 1)


def _align_batches(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align batch dimension by slicing to min B."""
    if a.ndim != 4 or b.ndim != 4:
        return a, b
    ba = a.shape[0]
    bb = b.shape[0]
    if ba == bb:
        return a, b
    m = min(ba, bb)
    return a[:m], b[:m]


def apply_channel_ops(
    image: torch.Tensor,
    operation: str,
    source: str,
    dest: str,
    amount_255: float,
    source_image: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply channel operations over RGB/HSV/OKLAB.

    image: [B,H,W,3] float in [0,1]
    operation: invert|overwrite|set|add|subtract|multiply|divide|clamp_max|clamp_min|truncate|contrast
    source: R|G|B|H|S|V|RGB|HSV|OKLAB (OKLAB applies to all three Oklab channels)
    dest: R|G|B|H|S|V (only used for overwrite)
    amount_255: amount in 0..255 scale
    """
    # Normalize operation label from UI (e.g., "Clamp Max" -> "clamp_max")
    op = operation.strip().lower().replace(" ", "_")

    rgb = image

    # Normalize full-name channels to internal short codes
    name_map = {
        "RED": "R", "GREEN": "G", "BLUE": "B",
        "HUE": "H", "SATURATION": "S", "VALUE": "V",
        "RGB": "RGB", "HSV": "HSV",
        "OKLAB": "OKLAB",
    }
    s = name_map.get(source.upper(), source.upper())
    d = name_map.get(dest.upper(), dest.upper())

    if op == "invert":
        if s == "RGB":
            rgb = 1.0 - rgb
        elif s in ("R", "G", "B"):
            idx = {"R": 0, "G": 1, "B": 2}[s]
            rgb = rgb.clone()
            rgb[..., idx] = 1.0 - rgb[..., idx]
        elif s in ("H", "S", "V", "HSV"):
            hsv = rgb_to_hsv(rgb)
            if s == "HSV":
                hsv = 1.0 - hsv
            else:
                idx = {"H": 0, "S": 1, "V": 2}[s]
                hsv = hsv.clone()
                hsv[..., idx] = 1.0 - hsv[..., idx]
            rgb = hsv_to_rgb(hsv)
        elif s == "OKLAB":
            lab = rgb_to_oklab(rgb)
            L = 1.0 - torch.clamp(lab[..., 0:1], 0.0, 1.0)
            a = lab[..., 1:2]; bch = lab[..., 2:3]
            a_n = (a + 0.4) / 0.8; b_n = (bch + 0.4) / 0.8
            a_i = (1.0 - torch.clamp(a_n, 0.0, 1.0)) * 0.8 - 0.4
            b_i = (1.0 - torch.clamp(b_n, 0.0, 1.0)) * 0.8 - 0.4
            lab = torch.cat([L, a_i, b_i], dim=-1)
            rgb = oklab_to_rgb(lab)

    elif op == "overwrite" or op == "overwrite_from_image":
        # For overwrite_from_image, take source value from source_image if provided;
        # otherwise behave like normal overwrite.
        rgb = rgb.clone()
        src_rgb = rgb
        if op == "overwrite_from_image" and source_image is not None:
            # Align batch and spatial dims
            src_rgb = source_image
            src_rgb, rgb = _align_batches(src_rgb, rgb)
            # Ensure src spatial matches dest
            _, th, tw, _ = rgb.shape
            src_rgb = _resize_hw(src_rgb, (th, tw))

        sval = None
        if s in ("R", "G", "B"):
            idx = {"R": 0, "G": 1, "B": 2}[s]
            sval = src_rgb[..., idx:idx+1]
        elif s in ("H", "S", "V") or s == "HSV":
            hsv = rgb_to_hsv(src_rgb)
            if s == "HSV":
                sval = hsv[..., 0:1]
            else:
                idx = {"H": 0, "S": 1, "V": 2}[s]
                sval = hsv[..., idx:idx+1]
        elif s == "OKLAB":
            lab = rgb_to_oklab(src_rgb)
            sval = lab[..., 0:1]
        else:
            sval = src_rgb[..., 0:1]

        if d in ("R", "G", "B"):
            idx = {"R": 0, "G": 1, "B": 2}[d]
            rgb[..., idx:idx+1] = torch.clamp(sval, 0.0, 1.0)
        elif d in ("H", "S", "V"):
            hsv = rgb_to_hsv(rgb)
            idx = {"H": 0, "S": 1, "V": 2}[d]
            hsv[..., idx:idx+1] = torch.clamp(sval, 0.0, 1.0)
            rgb = hsv_to_rgb(hsv)

    else:
        amount_raw = float(amount_255)
        amt_norm = amount_raw / 255.0
        if op in ("set", "add", "subtract", "multiply", "divide", "clamp_max", "clamp_min", "truncate", "contrast"):
            if op in ("multiply", "divide"):
                amt_eff = amount_raw
            elif op == "truncate":
                amt_eff = 1.0 - amt_norm
            elif op == "contrast":
                amt_eff = 1.0 - amt_norm
            else:
                amt_eff = amt_norm
            if s == "RGB":
                rgb = _apply_op(rgb, op, amt_eff)
                rgb = torch.clamp(rgb, 0.0, 1.0)
            elif s in ("R", "G", "B"):
                idx = {"R": 0, "G": 1, "B": 2}[s]
                rgb = rgb.clone()
                rgb[..., idx] = torch.clamp(_apply_op(rgb[..., idx], op, amt_eff), 0.0, 1.0)
            elif s in ("H", "S", "V", "HSV"):
                hsv = rgb_to_hsv(rgb)
                if s == "HSV":
                    hsv = torch.stack([
                        torch.remainder(_apply_op(hsv[..., 0], op, amt_eff), 1.0),
                        torch.clamp(_apply_op(hsv[..., 1], op, amt_eff), 0.0, 1.0),
                        torch.clamp(_apply_op(hsv[..., 2], op, amt_eff), 0.0, 1.0),
                    ], dim=-1)
                else:
                    idx = {"H": 0, "S": 1, "V": 2}[s]
                    if idx == 0:
                        hsv[..., 0] = torch.remainder(_apply_op(hsv[..., 0], op, amt_eff), 1.0)
                    else:
                        hsv[..., idx] = torch.clamp(_apply_op(hsv[..., idx], op, amt_eff), 0.0, 1.0)
                rgb = hsv_to_rgb(hsv)
            elif s == "OKLAB":
                lab = rgb_to_oklab(rgb)
                L = torch.clamp(_apply_op(lab[..., 0], op, amt_eff), 0.0, 1.0)
                a = lab[..., 1]; bch = lab[..., 2]
                if op in ("set", "contrast", "truncate"):
                    a_n = (a + 0.4) / 0.8; b_n = (bch + 0.4) / 0.8
                    a_n = torch.clamp(_apply_op(a_n, op, amt_eff), 0.0, 1.0)
                    b_n = torch.clamp(_apply_op(b_n, op, amt_eff), 0.0, 1.0)
                    a2 = a_n * 0.8 - 0.4; b2 = b_n * 0.8 - 0.4
                else:
                    a2 = _apply_op(a, op, amt_eff); b2 = _apply_op(bch, op, amt_eff)
                lab = torch.stack([L, a2, b2], dim=-1)
                rgb = oklab_to_rgb(lab)

    return rgb


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


def _save_web_preview(img: torch.Tensor, web_dir: str, filename: str = "channel_ops_preview.png", max_dim: int = 512) -> None:
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
        print(f"[ChannelOpsNode] Preview save failed: {e}")


class ChannelOpsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": ([
                    "Invert",
                    "Overwrite",
                    "Overwrite from Image",
                    "Set",
                    "Add",
                    "Subtract",
                    "Multiply",
                    "Divide",
                    "Clamp Max",
                    "Clamp Min",
                    "Truncate",
                    "Contrast",
                ], {"default": "Invert"}),
                "Source": ([
                    "Red", "Green", "Blue",
                    "Hue", "Saturation", "Value",
                    "RGB", "HSV", "Oklab"
                ], {"default": "RGB"}),
                "Destination": ([
                    "Red", "Green", "Blue",
                    "Hue", "Saturation", "Value"
                ], {"default": "Red"}),
            },
            "optional": {
                "image_b": ("IMAGE",),
                "amount": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "preview_id": ("STRING", {"default": "A"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "preview_ref")
    FUNCTION = "run"
    CATEGORY = "image/processing"

    def run(self, image, operation, Source, Destination, amount=0, preview_id: str = "A", image_b=None):
        out = apply_channel_ops(image, operation, Source, Destination, amount, source_image=image_b)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        web_dir = os.path.join(this_dir, "web")
        safe_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in (preview_id or 'A'))
        filename = f"channel_ops_preview_{safe_id}.png"
        _save_web_preview(image, web_dir, filename=filename, max_dim=512)
        # Save secondary preview if provided (used for overwrite-from-image in frontend)
        if image_b is not None:
            filename_src = f"channel_ops_preview_src_{safe_id}.png"
            _save_web_preview(image_b, web_dir, filename=filename_src, max_dim=512)
        try:
            from server import PromptServer  # type: ignore
            payload = {
                "preview_id": safe_id,
                "filename": filename,
                "ts": time.time(),
            }
            PromptServer.instance.send_sync("channel_ops_preview", payload)
        except Exception:
            pass
        prev_ref = _downscale_for_preview(image, max_dim=512)
        return (out, prev_ref)
