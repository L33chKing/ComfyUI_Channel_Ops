![preview](./preview.gif)

# ComfyUI Channel Ops (live-preview)

Comfyui custom node that modifies image channels with various per-channel operations across RGB, HSV, and Oklab color-space with live in-node preview

Note: Destination parameter has no effect on any operations other than Overwrite

## Operations

| Operation | Effect |
|-----------|--------|
| Invert    | Inverts selected channel(s). |
| Overwrite | Overwrites destination channel with source channel. |
| Set       | Sets selected channel(s) to Amount. |
| Add       | Adds Amount. |
| Subtract  | Subtracts Amount. |
| Multiply  | Multiplies by factor. |
| Divide    | Divides by factor. |
| Clamp Min | Clamps selected channel(s) to be ≥ Amount. |
| Clamp Max | Clamps selected channel(s) to be ≤ Amount. |
| Truncate  | Quantizes using a step derived from Amount. |
| Contrast  | Adjusts contrast around mid-gray based on Amount. |

## Sources

| Source | Channels |
|--------|----------|
| RGB    | R, G, B  |
| R      | Red      |
| G      | Green    |
| B      | Blue     |
| HSV    | H, S, V  |
| H      | Hue      |
| S      | Saturation |
| V      | Value    |
| Oklab  | L, a, b  |

after node execution, the node saves a preview PNG and pushes a refresh for instant display.

if you find mismatch between preview and output image, 
make issue on repo.

MIT