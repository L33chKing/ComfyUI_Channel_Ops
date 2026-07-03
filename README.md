# ComfyUI Channel Ops (live-preview)

Four custom nodes for per-channel image manipulation, layer blending, image filters, and color matching — all with in-node live preview and optional MASK input.

## Channel Ops

![Channel Ops preview](./channel_ops.gif)

Per-channel operations across RGB / HSV / Oklab. Destination only affects Overwrite ops. `mask` input restricts the effect to the masked region.

| Operations | Sources / Destinations |
|---|---|
| Invert · Set · Add · Subtract · Multiply · Divide | Red · Green · Blue |
| Clamp Min · Clamp Max · Truncate · Contrast | Red+Green · Red+Blue · Green+Blue |
| Overwrite · Overwrite from Image | Hue · Saturation · Value · RGB · HSV · Oklab |

---

## Layer Blending Ops

![Layer Blending Ops preview](./Layer_blend.gif)

Composites two images with a blend mode + opacity (0–255). Foreground bilinearly resized to background H×W; batches align to `min(B)`; output clamped to [0, 1].

| Blend modes |
|---|
| Normal · Multiply · Screen · Overlay |
| Soft Light · Hard Light · Darken · Lighten |
| Color Dodge · Color Burn · Linear Dodge (Add) · Linear Burn |
| Vivid Light · Linear Light · Pin Light · Hard Mix |
| Difference · Exclusion · Add · Subtract · Divide |

---

## Filter Ops

![Filter Ops preview](./filter_ops.gif)

Adaptive widget UI — only the controls relevant to the current effect are shown. `effect` + `blend_mode` always visible. `mask` input restricts the effect to the masked region. Custom widgets (color wheels, histogram, vignette center) mirror the underlying numeric values both ways.

| Effect | Key params |
|---|---|
| **Blur** | `blur_mode` (Gaussian / Average / Edge Average), `radius`, `source` |
| **Halftone** | `shape` (12 patterns), `size`, `angle` + angle wheel, `contrast`, `brightness`, `inverse` — Bayer-dithered |
| **Color Filter** | `hue`, `density`, `preserve_highlights` + HSV color wheel |
| **Sharpen** | `amount` (negative = soften), `radius`, `threshold` |
| **Laplacian Sharpen** | `amount`, `kernel` (3×3 4-neighbor / 8-neighbor) — edge boost via the Laplacian operator |
| **Unsharp Masking** | `amount` (multiplier), `radius` (Gaussian), `threshold` — classic [unsharp mask](https://en.wikipedia.org/wiki/Unsharp_masking) |
| **Pixelate** | `size`, `mode` (Pixelate / Mosaic adds grout lines) |
| **Posterize** | `levels`, `mode` (RGB / Luminance), `dither_mode` (None / Bayer / Random / Floyd-Steinberg / Atkinson), `dither` strength |
| **Vignette** | `amount`, `size`, `feather`, `roundness` + draggable center widget |
| **Levels** | `channel`, in/out black/white, `gamma` + histogram widget with B/γ/W handles |
| **Color Balance** | 3 color wheels (Shadows / Midtones / Highlights); hue = tint direction, distance = intensity |

---

## Color Matching Ops

![Color Matching Ops preview](./color_match.gif)

Recolors `image` so its color statistics match a `reference` image. Two image inputs, one image output. Inputs may be any size — matching is statistics-only, so no resizing is needed.

| `method` | What it does |
|---|---|
| **LAB** *(default)* | Mean/std (Reinhard) transfer in the perceptual Oklab space |
| **RGB** | Mean/std transfer applied directly per RGB channel |
| **Histogram** | Per-channel cumulative-distribution (CDF) matching |

| `target` | What it keeps (split in Oklab; rest comes from the original) |
|---|---|
| **All** *(default)* | Full color match |
| **Lightness** | Match the reference tonality, keep the original colors |
| **Color** | Match the reference grade, keep the original tonality |

---

## License
MIT
