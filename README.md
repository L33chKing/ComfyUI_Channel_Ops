![preview](./preview.gif)

# ComfyUI Channel Ops (live-preview)

Three custom nodes for per-channel image manipulation, layer blending, and image filters — all with in-node live preview and optional MASK input.

## Channel Ops

Per-channel operations across RGB / HSV / Oklab. Destination only affects Overwrite ops. `mask` input restricts the effect to the masked region.

| Operations | Sources / Destinations |
|---|---|
| Invert · Set · Add · Subtract · Multiply · Divide | Red · Green · Blue |
| Clamp Min · Clamp Max · Truncate · Contrast | Red+Green · Red+Blue · Green+Blue |
| Overwrite · Overwrite from Image | Hue · Saturation · Value · RGB · HSV · Oklab |

---

## Layer Blending

Composites two images with a blend mode + opacity (0–255). Foreground bilinearly resized to background H×W; batches align to `min(B)`; output clamped to [0, 1].

| Blend modes |
|---|
| Normal · Multiply · Screen · Overlay |
| Soft Light · Hard Light · Darken · Lighten |
| Color Dodge · Color Burn · Linear Dodge (Add) · Linear Burn |
| Vivid Light · Linear Light · Pin Light · Hard Mix |
| Difference · Exclusion · Add · Subtract · Divide |
| Darker Color · Lighter Color |

---

## Filter Ops (live-preview)

Adaptive widget UI — only the controls relevant to the current effect are shown. `effect` + `blend_mode` always visible. `mask` input restricts the effect to the masked region. Custom widgets (color wheels, histogram, vignette center) mirror the underlying numeric values both ways.

| Effect | Key params |
|---|---|
| **Blur** | `blur_mode` (Gaussian / Average / Edge Average), `radius`, `source` |
| **Halftone** | `shape` (12 patterns), `size`, `angle` + angle wheel, `contrast`, `brightness`, `inverse` — Bayer-dithered |
| **Color Filter** | `hue`, `density`, `preserve_highlights` + HSV color wheel |
| **Sharpen** | `amount` (negative = soften), `radius`, `threshold` |
| **Pixelate** | `size`, `mode` (Pixelate / Mosaic adds grout lines) |
| **Posterize** | `levels`, `mode` (RGB / Luminance), `dither_mode` (None / Bayer / Random / Floyd-Steinberg / Atkinson), `dither` strength |
| **Vignette** | `amount`, `size`, `feather`, `roundness` + draggable center widget |
| **Levels** | `channel`, in/out black/white, `gamma` + histogram widget with B/γ/W handles |
| **Color Balance** | 3 color wheels (Shadows / Midtones / Highlights); hue = tint direction, distance = intensity |

---

## License
MIT
