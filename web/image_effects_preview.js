// Image Effects - In-node live preview (canvas) + dynamic widgets.
// - Blur effects (Average / Average Edge / Gaussian) use radius + source widgets.
// - Halftone uses its own shape/inverse/size/angle/contrast/brightness/quality widgets.
// - Effect dropdown switches which widget set is visible; values for hidden widgets
//   are kept (serialized) so flipping back to a previous effect remembers settings.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const EXT_NAME = "Image_Effects.Preview";

(function(){
  // ---------- which widgets belong to which effect ----------
  // Old workflows may carry these effect names — they're aliased to the new
  // single "Blur" effect via blur_mode during the onConfigure migration below.
  const BLUR_EFFECTS = new Set(['blur', 'average blur', 'average edge blur', 'gaussian blur']);
  const BLUR_WIDGETS = ['radius', 'source', 'blur_mode'];

  // Maps the new blur_mode dropdown values to the legacy effect-name strings
  // that the JS-side blur primitives still dispatch by.
  function resolveBlurEffectName(blurMode){
    const m = String(blurMode || 'Gaussian').toLowerCase();
    if(m === 'average') return 'average blur';
    if(m === 'edge average') return 'average edge blur';
    return 'gaussian blur';
  }
  const HALFTONE_WIDGETS = [
    'halftone_shape', 'halftone_inverse', 'halftone_size',
    'halftone_angle', 'halftone_angle_wheel',
    'halftone_contrast', 'halftone_brightness',
    // halftone_quality and halftone_dither are intentionally NOT in this list.
    // Quality is fixed at 3 and dither is fixed at 100; both widgets stay in
    // INPUT_TYPES for positional back-compat but are always hidden.
  ];
  const COLOR_FILTER_WIDGETS = [
    'color_hue', 'color_density', 'color_preserve_highlights', 'color_wheel',
  ];
  const SHARPEN_WIDGETS = ['sharpen_amount', 'sharpen_radius', 'sharpen_threshold'];
  const LAPLACIAN_WIDGETS = ['laplacian_amount', 'laplacian_kernel'];
  const UNSHARP_WIDGETS = ['usm_amount', 'usm_radius', 'usm_threshold'];
  const PIXELATE_WIDGETS = ['pixelate_size', 'pixelate_mode'];
  const POSTERIZE_WIDGETS = ['posterize_levels', 'posterize_mode', 'posterize_dither_mode', 'posterize_dither'];
  const VIGNETTE_WIDGETS = [
    'vignette_widget',
    'vignette_amount', 'vignette_size', 'vignette_feather', 'vignette_roundness',
    // center_x / center_y are controlled exclusively by the custom widget;
    // their sliders stay in INPUT_TYPES for serialisation but are always hidden.
  ];
  const LEVELS_WIDGETS = [
    'levels_channel', 'levels_widget',
    'levels_in_black', 'levels_in_white', 'levels_gamma',
    'levels_out_black', 'levels_out_white',
  ];
  // The 9 R/G/B sliders are kept in INPUT_TYPES for serialisation but are
  // always hidden — the custom cb_widget is the only visible control.
  const COLOR_BALANCE_WIDGETS = ['cb_widget'];
  const COLOR_BALANCE_HIDDEN_SLIDERS = [
    'cb_shadow_red', 'cb_shadow_green', 'cb_shadow_blue',
    'cb_midtone_red', 'cb_midtone_green', 'cb_midtone_blue',
    'cb_highlight_red', 'cb_highlight_green', 'cb_highlight_blue',
  ];
  // Used by the workflow-load migration: detect old positional layout where
  // blend_mode lived at index 3, not 1.
  const BLEND_MODE_NAMES = new Set([
    'Normal','Multiply','Screen','Overlay','Soft Light','Hard Light',
    'Darken','Lighten','Color Dodge','Color Burn',
    'Linear Dodge (Add)','Linear Burn','Vivid Light','Linear Light','Pin Light',
    'Difference','Exclusion','Add','Subtract','Divide','Hard Mix',
  ]);

  // Walk node.widgets in order and return the Y coord where `target` starts.
  // This is a reliable alternative to widget.last_y for widgets that may not
  // have been drawn yet (e.g. a wheel just made visible by an effect change).
  function computeWidgetY(node, target){
    if(!node || !node.widgets) return 0;
    let y = (node.widgets_start_y ?? 30);
    for(const w of node.widgets){
      if(w === target) return y;
      if(!w || w.hidden) continue;
      let h = 26;
      try{
        if(typeof w.computeSize === 'function'){
          const sz = w.computeSize(node.size[0]);
          if(Array.isArray(sz)) h = sz[1] || 0;
        }
      }catch(_){ }
      if(h > 0) y += h + 4;
    }
    return y;
  }

  // ---------- color-space conversions ----------
  function rgbToHsvPlanes(R, G, B, outH, outS, outV){
    // Standard RGB→HSV: matches channel_ops.rgb_to_hsv in Python so the
    // HSV/Hue/Sat/Value source paths produce the same colors as the backend.
    const n = R.length;
    for(let i=0; i<n; i++){
      const r = R[i], g = G[i], b = B[i];
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      const d = mx - mn;
      outV[i] = mx;
      outS[i] = mx > 0 ? d / (mx + 1e-8) : 0;
      let h = 0;
      if(d > 1e-8){
        if(mx === r) h = (((g - b) / d) / 6.0) % 1.0;
        else if(mx === g) h = ((2.0 + (b - r) / d) / 6.0) % 1.0;
        else h = ((4.0 + (r - g) / d) / 6.0) % 1.0;
        if(h < 0) h += 1.0;
      }
      outH[i] = h;
    }
  }
  function hsvToRgbPlanes(H, S, V, outR, outG, outB){
    const n = H.length;
    for(let i=0; i<n; i++){
      const h = H[i], s = S[i], v = V[i];
      const h6 = h * 6.0;
      const ii = Math.floor(h6);
      const f = h6 - ii;
      const p = v * (1.0 - s);
      const q = v * (1.0 - s * f);
      const t = v * (1.0 - s * (1.0 - f));
      let r, g, b;
      switch(((ii % 6) + 6) % 6){
        case 0: r=v; g=t; b=p; break;
        case 1: r=q; g=v; b=p; break;
        case 2: r=p; g=v; b=t; break;
        case 3: r=p; g=q; b=v; break;
        case 4: r=t; g=p; b=v; break;
        default: r=v; g=p; b=q;
      }
      outR[i] = r < 0 ? 0 : (r > 1 ? 1 : r);
      outG[i] = g < 0 ? 0 : (g > 1 ? 1 : g);
      outB[i] = b < 0 ? 0 : (b > 1 ? 1 : b);
    }
  }
  function srgbToLinear(c){ return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function linearToSrgb(c){ return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(0, c), 1.0 / 2.4) - 0.055; }
  function rgbToOklabPlanes(R, G, B, outL, outA, outBl){
    const n = R.length;
    for(let i=0; i<n; i++){
      const r = srgbToLinear(R[i]);
      const g = srgbToLinear(G[i]);
      const b = srgbToLinear(B[i]);
      const l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b;
      const m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b;
      const s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b;
      const l_ = Math.cbrt(l), m_ = Math.cbrt(m), s_ = Math.cbrt(s);
      outL[i]  = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_;
      outA[i]  = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_;
      outBl[i] = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_;
    }
  }
  function oklabToRgbPlanes(L, A, Bl, outR, outG, outB){
    const n = L.length;
    for(let i=0; i<n; i++){
      const l_ = L[i] + 0.3963377774*A[i] + 0.2158037573*Bl[i];
      const m_ = L[i] - 0.1055613458*A[i] - 0.0638541728*Bl[i];
      const s_ = L[i] - 0.0894841775*A[i] - 1.2914855480*Bl[i];
      const l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
      const r =  4.0767416621*l - 3.3077115913*m + 0.2309699292*s;
      const g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s;
      const b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s;
      let rs = linearToSrgb(r), gs = linearToSrgb(g), bs = linearToSrgb(b);
      outR[i] = rs < 0 ? 0 : (rs > 1 ? 1 : rs);
      outG[i] = gs < 0 ? 0 : (gs > 1 ? 1 : gs);
      outB[i] = bs < 0 ? 0 : (bs > 1 ? 1 : bs);
    }
  }

  // ---------- blur primitives on a planar Float32 plane ----------
  function boxBlurPlane(s, tmp, o, w, h, radius){
    if(radius <= 0){ o.set(s); return; }
    const r = radius|0;
    const k = 2*r + 1;
    const invK = 1.0 / k;
    for(let yy=0; yy<h; yy++){
      const row = yy*w;
      let sum = 0;
      for(let dx=-r; dx<=r; dx++){
        const sx = dx < 0 ? 0 : (dx >= w ? w-1 : dx);
        sum += s[row + sx];
      }
      tmp[row] = sum * invK;
      for(let xx=1; xx<w; xx++){
        const ax = (xx + r) >= w ? w-1 : (xx + r);
        const rx = (xx - r - 1) < 0 ? 0 : (xx - r - 1);
        sum += s[row + ax] - s[row + rx];
        tmp[row + xx] = sum * invK;
      }
    }
    for(let xx=0; xx<w; xx++){
      let sum = 0;
      for(let dy=-r; dy<=r; dy++){
        const sy = dy < 0 ? 0 : (dy >= h ? h-1 : dy);
        sum += tmp[sy*w + xx];
      }
      o[xx] = sum * invK;
      for(let yy=1; yy<h; yy++){
        const ay = (yy + r) >= h ? h-1 : (yy + r);
        const ry = (yy - r - 1) < 0 ? 0 : (yy - r - 1);
        sum += tmp[ay*w + xx] - tmp[ry*w + xx];
        o[yy*w + xx] = sum * invK;
      }
    }
  }
  function bresenhamCircleOffsets(r){
    if(r <= 0) return [[0,0]];
    const seen = new Set();
    const out = [];
    let x = 0, y = r|0, d = 1 - (r|0);
    while(x <= y){
      const pts = [
        [x,y],[-x,y],[x,-y],[-x,-y],
        [y,x],[-y,x],[y,-x],[-y,-x],
      ];
      for(const p of pts){
        const k = p[0]+','+p[1];
        if(!seen.has(k)){ seen.add(k); out.push(p); }
      }
      x++;
      if(d < 0) d += 2*x + 1;
      else { y--; d += 2*(x-y) + 1; }
    }
    return out;
  }
  function averageEdgeBlurPlane(s, o, w, h, radius){
    if(radius <= 0){ o.set(s); return; }
    const offs = bresenhamCircleOffsets(radius|0);
    const n = offs.length;
    const invN = 1.0 / n;
    for(let yy=0; yy<h; yy++){
      for(let xx=0; xx<w; xx++){
        let acc = 0;
        for(let i=0; i<n; i++){
          let sx = xx + offs[i][0];
          let sy = yy + offs[i][1];
          if(sx<0) sx=0; else if(sx>=w) sx=w-1;
          if(sy<0) sy=0; else if(sy>=h) sy=h-1;
          acc += s[sy*w + sx];
        }
        o[yy*w + xx] = acc * invN;
      }
    }
  }
  function gaussianBlurPlane(s, tmp, o, w, h, radiusParam){
    if(radiusParam <= 0){ o.set(s); return; }
    const sigma = Math.max(0.5, radiusParam / 3.0);
    const kr = Math.max(1, Math.round(3 * sigma));
    const k = new Float32Array(2*kr + 1);
    let ksum = 0;
    for(let i=-kr; i<=kr; i++){
      const v = Math.exp(-(i*i) / (2*sigma*sigma));
      k[i+kr] = v; ksum += v;
    }
    for(let i=0; i<k.length; i++) k[i] /= ksum;
    for(let yy=0; yy<h; yy++){
      const row = yy*w;
      for(let xx=0; xx<w; xx++){
        let acc = 0;
        for(let i=-kr; i<=kr; i++){
          let sx = xx + i;
          if(sx<0) sx=0; else if(sx>=w) sx=w-1;
          acc += s[row + sx] * k[i+kr];
        }
        tmp[row + xx] = acc;
      }
    }
    for(let xx=0; xx<w; xx++){
      for(let yy=0; yy<h; yy++){
        let acc = 0;
        for(let i=-kr; i<=kr; i++){
          let sy = yy + i;
          if(sy<0) sy=0; else if(sy>=h) sy=h-1;
          acc += tmp[sy*w + xx] * k[i+kr];
        }
        o[yy*w + xx] = acc;
      }
    }
  }
  function applyEffectPlane(effect, s, tmp, o, w, h, radius){
    if(effect === 'average blur') boxBlurPlane(s, tmp, o, w, h, radius);
    else if(effect === 'average edge blur') averageEdgeBlurPlane(s, o, w, h, radius);
    else if(effect === 'gaussian blur') gaussianBlurPlane(s, tmp, o, w, h, radius);
    else o.set(s);
  }
  function blurHuePlane(srcH, tmp, out, cosBuf, sinBuf, cosBlurBuf, sinBlurBuf, w, h, effect, radius){
    const twoPi = Math.PI * 2;
    const n = srcH.length;
    for(let i=0; i<n; i++){
      cosBuf[i] = Math.cos(srcH[i] * twoPi);
      sinBuf[i] = Math.sin(srcH[i] * twoPi);
    }
    applyEffectPlane(effect, cosBuf, tmp, cosBlurBuf, w, h, radius);
    applyEffectPlane(effect, sinBuf, tmp, sinBlurBuf, w, h, radius);
    for(let i=0; i<n; i++){
      let hv = Math.atan2(sinBlurBuf[i], cosBlurBuf[i]) / twoPi;
      if(hv < 0) hv += 1;
      out[i] = hv;
    }
  }

  // ---------- halftone ----------
  // 8x8 Bayer matrix (values 0..63), divided by 64 -> [0, 1).
  const BAYER_8 = [
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
  ];

  // Classic halftone: per-pixel comparison of luminance against a shape-
  // dependent "screen value". Ordered Bayer dithering on the luminance side
  // produces real noisy/error-diffusion-like tonal transitions.
  function applyHalftone(sR, sG, sB, w, h, params, tmpPlane, outR, outG, outB){
    const shape = String(params.shape || 'Dot').toLowerCase();
    const inverse = !!params.inverse;
    const sz = Math.max(1, params.size|0);
    const angleRad = (params.angle || 0) * Math.PI / 180;
    const cosA = Math.cos(angleRad), sinA = Math.sin(angleRad);
    const dStrength = Math.max(0, Math.min(100, params.dither|0)) / 100;
    const bayerCell = Math.max(1, (sz / 5) | 0);
    const cxImg = (w - 1) / 2;
    const cyImg = (h - 1) / 2;
    const bOff = (params.brightness || 0) / 100.0;
    const cFac = ((params.contrast || 0) + 100.0) / 100.0;
    const N = w * h;
    const invSz = 1 / sz;

    // Per-pixel adjusted luminance.
    const lum = tmpPlane;
    for(let p=0; p<N; p++){
      let r = (sR[p] - 0.5) * cFac + 0.5 + bOff;
      let g = (sG[p] - 0.5) * cFac + 0.5 + bOff;
      let b = (sB[p] - 0.5) * cFac + 0.5 + bOff;
      if(r<0)r=0; else if(r>1)r=1;
      if(g<0)g=0; else if(g>1)g=1;
      if(b<0)b=0; else if(b>1)b=1;
      lum[p] = 0.299*r + 0.587*g + 0.114*b;
    }

    // Compute screen value + threshold per pixel.
    for(let y=0; y<h; y++){
      const ys = y - cyImg;
      for(let x=0; x<w; x++){
        const xs = x - cxImg;
        const rx = (xs * cosA + ys * sinA) * invSz;
        const ry = (-xs * sinA + ys * cosA) * invSz;
        const cxCell = Math.floor(rx);
        const cyCell = Math.floor(ry);
        const lx = rx - cxCell - 0.5;
        const ly = ry - cyCell - 0.5;

        // Screen value for the current shape (high in dense area, low elsewhere).
        let d, screen;
        switch(shape){
          case 'dot': case 'round': case 'spot':
            d = Math.sqrt(lx*lx + ly*ly) / 0.7071;
            screen = 1 - Math.min(1, Math.max(0, d));
            break;
          case 'square dot':
            d = 2 * Math.max(Math.abs(lx), Math.abs(ly));
            screen = 1 - Math.min(1, Math.max(0, d));
            break;
          case 'line':
          case 'line centered':
            screen = 1 - Math.min(1, Math.max(0, 2 * Math.abs(ly)));
            break;
          case 'line scalloped': {
            const wave = Math.sin(rx * Math.PI * 2) * 0.15;
            screen = 1 - Math.min(1, Math.max(0, 2 * Math.abs(ly - wave)));
            break;
          }
          case 'rhomboid':
          case 'spot diamond':
            d = 2 * (Math.abs(lx) + Math.abs(ly));
            screen = 1 - Math.min(1, Math.max(0, d));
            break;
          case 'cross cut':
            d = 2 * Math.min(Math.abs(lx), Math.abs(ly));
            screen = 1 - Math.min(1, Math.max(0, d));
            break;
          case 'saddle':
            d = 4 * Math.abs(lx * ly);
            screen = 1 - Math.min(1, Math.max(0, d));
            break;
          case 'random dots': {
            const seed = (Math.abs((cxCell|0) * 73856093 + (cyCell|0) * 19349663)) >>> 0;
            const dxJit = ((seed % 1000) / 1000 - 0.5) * 0.6;
            const dyJit = ((Math.floor(seed / 1000) % 1000) / 1000 - 0.5) * 0.6;
            d = Math.sqrt((lx-dxJit)*(lx-dxJit) + (ly-dyJit)*(ly-dyJit)) / 0.7071;
            screen = 1 - Math.min(1, Math.max(0, d));
            break;
          }
          default:
            d = Math.sqrt(lx*lx + ly*ly) / 0.7071;
            screen = 1 - Math.min(1, Math.max(0, d));
        }

        // Dithered luminance: Bayer noise added in stable ordered pattern.
        const p = y*w + x;
        let effLum = lum[p];
        if(dStrength > 0){
          const by = ((y / bayerCell) | 0) & 7;
          const bx = ((x / bayerCell) | 0) & 7;
          const bayer = BAYER_8[by][bx] / 64;
          effLum += (bayer - 0.5) * dStrength;
        }

        let result;
        if(dStrength > 0){
          result = effLum > screen ? 1 : 0;
        } else {
          // No dither: soft margin around the screen value so we don't
          // get hard-binary output without any tonal smoothing.
          result = Math.min(1, Math.max(0, (effLum - screen) * 4 + 0.5));
        }
        if(inverse) result = 1 - result;
        outR[p] = result; outG[p] = result; outB[p] = result;
      }
    }
  }

  // ---------- color filter ----------
  function hsvToRgbScalar(h, s, v){
    const h6 = h * 6;
    const ii = Math.floor(h6);
    const f = h6 - ii;
    const p = v * (1 - s);
    const q = v * (1 - s * f);
    const t = v * (1 - s * (1 - f));
    switch(((ii % 6) + 6) % 6){
      case 0: return [v, t, p];
      case 1: return [q, v, p];
      case 2: return [p, v, t];
      case 3: return [p, q, v];
      case 4: return [t, p, v];
      default: return [v, p, q];
    }
  }
  function applyColorFilter(sR, sG, sB, w, h, params, outR, outG, outB){
    const hueDeg = ((params.hue || 0) % 360 + 360) % 360;
    // Saturation is always 1.0 — the wheel only writes hue + density now.
    const density = Math.min(1, Math.max(0, (params.density || 0) / 255));
    const preserve = Math.min(1, Math.max(0, (params.preserve || 0) / 100));
    const [fr, fg, fb] = hsvToRgbScalar(hueDeg / 360, 1.0, 1.0);
    const N = w * h;
    for(let p=0; p<N; p++){
      const ar = sR[p], ag = sG[p], ab = sB[p];
      const lum = 0.299*ar + 0.587*ag + 0.114*ab;
      // Per-pixel effective density: highlights resist the tint based on preserve.
      let eff = density * (1 - preserve * lum);
      if(eff < 0) eff = 0; else if(eff > 1) eff = 1;
      let r = ar * (1 - eff) + (ar * fr) * eff;
      let g = ag * (1 - eff) + (ag * fg) * eff;
      let b = ab * (1 - eff) + (ab * fb) * eff;
      if(r<0)r=0; else if(r>1)r=1;
      if(g<0)g=0; else if(g>1)g=1;
      if(b<0)b=0; else if(b>1)b=1;
      outR[p] = r; outG[p] = g; outB[p] = b;
    }
  }

  // Pre-rendered HSV wheel cache (shared across all nodes).
  let _wheelCanvas = null;
  function getWheelCanvas(){
    if(_wheelCanvas) return _wheelCanvas;
    const sz = 192;
    const c = document.createElement('canvas');
    c.width = sz; c.height = sz;
    const cx = c.getContext('2d');
    const img = cx.createImageData(sz, sz);
    const cc = sz / 2;
    const rmax = sz / 2 - 1;
    for(let y=0; y<sz; y++){
      for(let x=0; x<sz; x++){
        const dx = x - cc, dy = y - cc;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const idx = (y*sz + x) * 4;
        if(dist > rmax){
          img.data[idx+3] = 0;
          continue;
        }
        // Standard color-wheel convention: 0deg = top (red), clockwise.
        // atan2(dx, -dy) gives 0 at up, +pi/2 at right, +pi at down.
        const ang = Math.atan2(dx, -dy);
        let hue = (ang * 180 / Math.PI + 360) % 360;
        const s = Math.min(1, dist / rmax);
        const [r, g, b] = hsvToRgbScalar(hue / 360, s, 1.0);
        img.data[idx]   = (r * 255) | 0;
        img.data[idx+1] = (g * 255) | 0;
        img.data[idx+2] = (b * 255) | 0;
        img.data[idx+3] = 255;
      }
    }
    cx.putImageData(img, 0, 0);
    _wheelCanvas = c;
    return c;
  }

  // ---------- sharpen (unsharp mask) ----------
  function applySharpen(sR, sG, sB, w, h, params, tmpPlane, outR, outG, outB){
    const amount = (params.amount || 0) / 100;
    if(Math.abs(amount) < 0.001){
      outR.set(sR); outG.set(sG); outB.set(sB); return;
    }
    const radius = Math.max(1, params.radius|0);
    const threshold = Math.max(0, (params.threshold|0)) / 255;
    // Gaussian-blur each plane into a temp buffer; we need 3 separate planes so
    // reuse tmpPlane for one and allocate two more locally.
    const blurR = new Float32Array(sR.length);
    const blurG = new Float32Array(sR.length);
    const blurB = new Float32Array(sR.length);
    gaussianBlurPlane(sR, tmpPlane, blurR, w, h, radius);
    gaussianBlurPlane(sG, tmpPlane, blurG, w, h, radius);
    gaussianBlurPlane(sB, tmpPlane, blurB, w, h, radius);
    const N = w * h;
    for(let p=0; p<N; p++){
      let dr = sR[p] - blurR[p];
      let dg = sG[p] - blurG[p];
      let db = sB[p] - blurB[p];
      if(threshold > 0){
        if(Math.abs(dr) <= threshold) dr = 0;
        if(Math.abs(dg) <= threshold) dg = 0;
        if(Math.abs(db) <= threshold) db = 0;
      }
      let r = sR[p] + dr * amount;
      let g = sG[p] + dg * amount;
      let b = sB[p] + db * amount;
      if(r<0)r=0; else if(r>1)r=1;
      if(g<0)g=0; else if(g>1)g=1;
      if(b<0)b=0; else if(b>1)b=1;
      outR[p] = r; outG[p] = g; outB[p] = b;
    }
  }

  // ---------- laplacian sharpen ----------
  // g = f - amount * Laplacian(f), replicate-padded 3x3 kernel (4- or 8-neighbor).
  function applyLaplacianSharpen(sR, sG, sB, w, h, params, outR, outG, outB){
    const amount = params.amount || 0;
    const use8 = String(params.kernel || '').indexOf('4') < 0;
    if(Math.abs(amount) < 0.001){
      outR.set(sR); outG.set(sG); outB.set(sB); return;
    }
    function plane(s, o){
      for(let y=0; y<h; y++){
        const ym = y > 0 ? y-1 : 0;
        const yp = y < h-1 ? y+1 : h-1;
        for(let x=0; x<w; x++){
          const xm = x > 0 ? x-1 : 0;
          const xp = x < w-1 ? x+1 : w-1;
          const c = s[y*w + x];
          let lap;
          if(use8){
            lap = s[ym*w+xm] + s[ym*w+x] + s[ym*w+xp]
                + s[y*w+xm]              + s[y*w+xp]
                + s[yp*w+xm] + s[yp*w+x] + s[yp*w+xp] - 8*c;
          } else {
            lap = s[ym*w+x] + s[y*w+xm] + s[y*w+xp] + s[yp*w+x] - 4*c;
          }
          let v = c - amount * lap;
          o[y*w + x] = v < 0 ? 0 : (v > 1 ? 1 : v);
        }
      }
    }
    plane(sR, outR); plane(sG, outG); plane(sB, outB);
  }

  // ---------- unsharp masking (Gaussian) ----------
  function applyUnsharpMask(sR, sG, sB, w, h, params, tmpPlane, outR, outG, outB){
    const amount = params.amount || 0;
    if(Math.abs(amount) < 0.001){
      outR.set(sR); outG.set(sG); outB.set(sB); return;
    }
    const radius = Math.max(1, params.radius|0);
    const threshold = Math.max(0, (params.threshold|0)) / 255;
    const blurR = new Float32Array(sR.length);
    const blurG = new Float32Array(sR.length);
    const blurB = new Float32Array(sR.length);
    gaussianBlurPlane(sR, tmpPlane, blurR, w, h, radius);
    gaussianBlurPlane(sG, tmpPlane, blurG, w, h, radius);
    gaussianBlurPlane(sB, tmpPlane, blurB, w, h, radius);
    const N = w * h;
    for(let p=0; p<N; p++){
      let dr = sR[p] - blurR[p];
      let dg = sG[p] - blurG[p];
      let db = sB[p] - blurB[p];
      if(threshold > 0){
        if(Math.abs(dr) <= threshold) dr = 0;
        if(Math.abs(dg) <= threshold) dg = 0;
        if(Math.abs(db) <= threshold) db = 0;
      }
      let r = sR[p] + dr * amount;
      let g = sG[p] + dg * amount;
      let b = sB[p] + db * amount;
      if(r<0)r=0; else if(r>1)r=1;
      if(g<0)g=0; else if(g>1)g=1;
      if(b<0)b=0; else if(b>1)b=1;
      outR[p] = r; outG[p] = g; outB[p] = b;
    }
  }

  // ---------- pixelate / mosaic ----------
  function applyPixelate(sR, sG, sB, w, h, params, outR, outG, outB){
    const bsize = Math.max(1, params.size|0);
    const mosaic = String(params.mode || 'Pixelate').toLowerCase() === 'mosaic';
    if(bsize === 1 && !mosaic){
      outR.set(sR); outG.set(sG); outB.set(sB); return;
    }
    // For each block: average then fill.
    for(let by=0; by<h; by+=bsize){
      const y2 = Math.min(h, by + bsize);
      for(let bx=0; bx<w; bx+=bsize){
        const x2 = Math.min(w, bx + bsize);
        let sumR=0, sumG=0, sumB=0, n=0;
        for(let yy=by; yy<y2; yy++){
          for(let xx=bx; xx<x2; xx++){
            const p = yy*w + xx;
            sumR += sR[p]; sumG += sG[p]; sumB += sB[p]; n++;
          }
        }
        const aR = sumR / n, aG = sumG / n, aB = sumB / n;
        for(let yy=by; yy<y2; yy++){
          for(let xx=bx; xx<x2; xx++){
            const p = yy*w + xx;
            // Mosaic: 1-pixel-wide dark grout at block edges (top + left).
            if(mosaic && bsize >= 3 && (yy === by || xx === bx)){
              outR[p] = aR * 0.3; outG[p] = aG * 0.3; outB[p] = aB * 0.3;
            } else {
              outR[p] = aR; outG[p] = aG; outB[p] = aB;
            }
          }
        }
      }
    }
  }

  // ---------- posterize ----------
  // In-place error-diffusion quantizer. `arr` is a Float32 plane modified in
  // place; for the JS preview we apply per-channel.
  function errorDiffusePlane(arr, w, h, step, kind){
    if(kind === 'floyd-steinberg'){
      for(let y=0; y<h; y++){
        for(let x=0; x<w; x++){
          const i = y*w + x;
          const old = arr[i];
          const ne = Math.round(old * step) / step;
          const err = old - ne;
          arr[i] = ne;
          if(x + 1 < w)              arr[i + 1]      += err * (7/16);
          if(y + 1 < h){
            if(x > 0)                arr[i + w - 1]  += err * (3/16);
                                     arr[i + w]      += err * (5/16);
            if(x + 1 < w)            arr[i + w + 1]  += err * (1/16);
          }
        }
      }
    } else { // atkinson
      for(let y=0; y<h; y++){
        for(let x=0; x<w; x++){
          const i = y*w + x;
          const old = arr[i];
          const ne = Math.round(old * step) / step;
          const err = (old - ne) / 8;
          arr[i] = ne;
          if(x + 1 < w)              arr[i + 1]      += err;
          if(x + 2 < w)              arr[i + 2]      += err;
          if(y + 1 < h){
            if(x > 0)                arr[i + w - 1]  += err;
                                     arr[i + w]      += err;
            if(x + 1 < w)            arr[i + w + 1]  += err;
          }
          if(y + 2 < h)              arr[i + 2*w]    += err;
        }
      }
    }
    // Clamp
    for(let i=0; i<arr.length; i++){
      if(arr[i] < 0) arr[i] = 0; else if(arr[i] > 1) arr[i] = 1;
    }
  }

  function applyPosterize(sR, sG, sB, w, h, params, outR, outG, outB){
    const l = Math.max(2, Math.min(64, params.levels|0));
    const step = l - 1;
    const mode = String(params.mode || 'RGB').toLowerCase();
    const dStrength = Math.max(0, Math.min(100, params.dither|0)) / 100;
    const dm = String(params.ditherMode || 'None').toLowerCase();
    const N = w * h;

    function offsetAt(x, y){
      if(dStrength <= 0) return 0;
      if(dm === 'bayer'){
        return (BAYER_8[y & 7][x & 7] / 64 - 0.5) * (dStrength / Math.max(1, step));
      }
      if(dm === 'random'){
        return (Math.random() - 0.5) * (dStrength / Math.max(1, step));
      }
      return 0;
    }
    function quantise(v){
      if(v < 0) v = 0; else if(v > 1) v = 1;
      return Math.round(v * step) / step;
    }

    // Error-diffusion paths copy src → out then mutate out in place.
    if(dm === 'floyd-steinberg' || dm === 'atkinson'){
      if(mode === 'luminance'){
        // Compute lum, error-diffuse it, then scale RGB.
        const lum = new Float32Array(N);
        for(let i=0; i<N; i++) lum[i] = 0.299*sR[i] + 0.587*sG[i] + 0.114*sB[i];
        const lumQ = new Float32Array(lum);
        errorDiffusePlane(lumQ, w, h, step, dm);
        for(let i=0; i<N; i++){
          const l_ = lum[i];
          const ratio = l_ > 1e-4 ? lumQ[i] / (l_ + 1e-8) : 1;
          let or = sR[i] * ratio, og = sG[i] * ratio, ob = sB[i] * ratio;
          if(or<0)or=0; else if(or>1)or=1;
          if(og<0)og=0; else if(og>1)og=1;
          if(ob<0)ob=0; else if(ob>1)ob=1;
          outR[i] = or; outG[i] = og; outB[i] = ob;
        }
      } else {
        outR.set(sR); outG.set(sG); outB.set(sB);
        errorDiffusePlane(outR, w, h, step, dm);
        errorDiffusePlane(outG, w, h, step, dm);
        errorDiffusePlane(outB, w, h, step, dm);
      }
      return;
    }

    // Ordered (Bayer) / Random / None — parallel per-pixel quantise.
    if(mode === 'luminance'){
      for(let y=0; y<h; y++){
        for(let x=0; x<w; x++){
          const p = y*w + x;
          const r = sR[p], g = sG[p], b = sB[p];
          const lum = 0.299*r + 0.587*g + 0.114*b;
          const lumQ = quantise(lum + offsetAt(x, y));
          const ratio = lum > 1e-4 ? lumQ / (lum + 1e-8) : 1;
          let or = r * ratio, og = g * ratio, ob = b * ratio;
          if(or<0)or=0; else if(or>1)or=1;
          if(og<0)og=0; else if(og>1)og=1;
          if(ob<0)ob=0; else if(ob>1)ob=1;
          outR[p] = or; outG[p] = og; outB[p] = ob;
        }
      }
      return;
    }
    for(let y=0; y<h; y++){
      for(let x=0; x<w; x++){
        const p = y*w + x;
        const o = offsetAt(x, y);
        outR[p] = quantise(sR[p] + o);
        outG[p] = quantise(sG[p] + o);
        outB[p] = quantise(sB[p] + o);
      }
    }
  }

  // ---------- vignette ----------
  function applyVignette(sR, sG, sB, w, h, params, outR, outG, outB){
    const amount = (params.amount || 0) / 100;
    const size = Math.max(0, (params.size || 0)) / 100;
    const feather = Math.max(0.001, (params.feather || 0) / 100);
    const rn = (params.roundness || 0) / 100;
    const cx_n = Math.max(0, Math.min(1, (params.cx == null ? 50 : params.cx) / 100));
    const cy_n = Math.max(0, Math.min(1, (params.cy == null ? 50 : params.cy) / 100));
    const cx = cx_n * Math.max(1, (w - 1));
    const cy = cy_n * Math.max(1, (h - 1));
    const halfW = Math.max(1, (w - 1) / 2);
    const halfH = Math.max(1, (h - 1) / 2);
    const outer = size + feather;
    for(let y=0; y<h; y++){
      for(let x=0; x<w; x++){
        let nx = (x - cx) / halfW;
        let ny = (y - cy) / halfH;
        if(rn > 0) ny *= (1 + rn);
        else if(rn < 0) nx *= (1 - rn);
        const dist = Math.sqrt(nx * nx + ny * ny);
        let f = (outer - dist) / Math.max(1e-6, outer - size);
        if(f < 0) f = 0; else if(f > 1) f = 1;
        f = f * f * (3 - 2 * f);
        const p = y * w + x;
        if(amount >= 0){
          const k = 1 - (1 - f) * amount;
          outR[p] = sR[p] * k;
          outG[p] = sG[p] * k;
          outB[p] = sB[p] * k;
        } else {
          const k = (1 - f) * (-amount);
          outR[p] = sR[p] + (1 - sR[p]) * k;
          outG[p] = sG[p] + (1 - sG[p]) * k;
          outB[p] = sB[p] + (1 - sB[p]) * k;
        }
      }
    }
  }

  // ---------- levels ----------
  function applyLevels(sR, sG, sB, w, h, params, outR, outG, outB){
    const inB = (params.inBlack || 0) / 255;
    const inW = (params.inWhite || 0) / 255;
    const outB_ = (params.outBlack || 0) / 255;
    const outW = (params.outWhite || 0) / 255;
    const g = Math.max(0.01, params.gamma || 1);
    const inRange = Math.max(1/255, inW - inB);
    const channel = String(params.channel || 'RGB').toUpperCase();

    function lvl(c){
      let x = (c - inB) / inRange;
      if(x < 0) x = 0; else if(x > 1) x = 1;
      x = Math.pow(x, 1 / g);
      x = outB_ + x * (outW - outB_);
      if(x < 0) x = 0; else if(x > 1) x = 1;
      return x;
    }

    const N = w * h;
    if(channel === 'RGB'){
      for(let p=0; p<N; p++){ outR[p] = lvl(sR[p]); outG[p] = lvl(sG[p]); outB[p] = lvl(sB[p]); }
    } else if(channel === 'RED' || channel === 'R'){
      for(let p=0; p<N; p++){ outR[p] = lvl(sR[p]); outG[p] = sG[p]; outB[p] = sB[p]; }
    } else if(channel === 'GREEN' || channel === 'G'){
      for(let p=0; p<N; p++){ outR[p] = sR[p]; outG[p] = lvl(sG[p]); outB[p] = sB[p]; }
    } else if(channel === 'BLUE' || channel === 'B'){
      for(let p=0; p<N; p++){ outR[p] = sR[p]; outG[p] = sG[p]; outB[p] = lvl(sB[p]); }
    } else {
      outR.set(sR); outG.set(sG); outB.set(sB);
    }
  }

  // ---------- color balance ----------
  function applyColorBalance(sR, sG, sB, w, h, params, outR, outG, outB){
    const s = 1 / 200;
    const sr = (params.sh_r || 0) * s, sg = (params.sh_g || 0) * s, sb = (params.sh_b || 0) * s;
    const mr = (params.mid_r || 0) * s, mg = (params.mid_g || 0) * s, mb = (params.mid_b || 0) * s;
    const hr = (params.hi_r || 0) * s, hg = (params.hi_g || 0) * s, hb = (params.hi_b || 0) * s;
    const N = w * h;
    for(let p=0; p<N; p++){
      const r = sR[p], g = sG[p], b = sB[p];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const shW = lum < 0.5 ? (1 - 2*lum) : 0;
      const midW = 1 - Math.abs(2*lum - 1);
      const hiW = lum > 0.5 ? (2*lum - 1) : 0;
      let or = r + sr*shW + mr*midW + hr*hiW;
      let og = g + sg*shW + mg*midW + hg*hiW;
      let ob = b + sb*shW + mb*midW + hb*hiW;
      if(or<0)or=0; else if(or>1)or=1;
      if(og<0)og=0; else if(og>1)og=1;
      if(ob<0)ob=0; else if(ob>1)ob=1;
      outR[p] = or; outG[p] = og; outB[p] = ob;
    }
  }

  // ---------- blend math ----------
  function blendPixel(a, b, mode){
    switch(mode){
      case 'multiply': return a*b;
      case 'screen': return 1 - (1-a)*(1-b);
      case 'overlay': return (a<=0.5) ? (2*a*b) : (1 - 2*(1-a)*(1-b));
      case 'hard light': return (b<=0.5) ? (2*a*b) : (1 - 2*(1-a)*(1-b));
      case 'soft light': return (1 - 2*b)*a*a + 2*b*a;
      case 'darken': return Math.min(a,b);
      case 'lighten': return Math.max(a,b);
      case 'color dodge': return (a / Math.max(1e-8, 1-b));
      case 'color burn': return (1 - ((1-a) / Math.max(1e-8, b)));
      case 'linear burn': return a + b - 1;
      case 'vivid light': return (b<0.5) ? (1 - ((1-a)/Math.max(1e-8, 2*b))) : (a/Math.max(1e-8, 2*(1-b)));
      case 'linear light': return a + 2*b - 1;
      case 'pin light': return (b<0.5) ? Math.min(a, Math.min(1, 2*b)) : Math.max(a, Math.max(0, 2*b-1));
      case 'difference': return Math.abs(a-b);
      case 'exclusion': return a + b - 2*a*b;
      case 'divide': return a / Math.max(1e-8, b);
      case 'hard mix': return (a + b < 1) ? 0 : 1;
      case 'linear dodge':
      case 'linear dodge (add)':
      case 'add': return a + b;
      case 'subtract': return a - b;
      case 'normal':
      default: return b;
    }
  }

  // ---------- dynamic widget visibility ----------
  function setHidden(w, hide){
    if(!w) return;
    w.hidden = !!hide;
    // Self-managing widgets (e.g. the custom color wheel) handle their own
    // computeSize based on `this.hidden` — leave their computeSize alone.
    if(w._keepOwnComputeSize) return;
    if(hide){
      if(w._origComputeSize === undefined) w._origComputeSize = w.computeSize || null;
      w.computeSize = () => [0, -4];
    } else {
      if(w._origComputeSize !== undefined){
        if(w._origComputeSize) w.computeSize = w._origComputeSize;
        else delete w.computeSize;
        delete w._origComputeSize;
      } else {
        delete w.computeSize;
      }
    }
  }
  function findWidget(node, name){
    if(!node || !node.widgets) return null;
    return node.widgets.find(w => w && (w.name === name || w.label === name)) || null;
  }
  function syncWidgets(node){
    if(!node || !node.widgets) return;
    const wEffect = findWidget(node, 'effect');
    const effect = String((wEffect && wEffect.value) || 'Average Blur').toLowerCase();
    const isBlur = BLUR_EFFECTS.has(effect);
    const isHalftone = effect === 'halftone';
    const isColorFilter = effect === 'color filter';
    const isSharpen = effect === 'sharpen';
    const isLaplacian = effect === 'laplacian sharpen';
    const isUnsharp = effect === 'unsharp masking';
    const isPixelate = effect === 'pixelate';
    const isPosterize = effect === 'posterize';
    const isVignette = effect === 'vignette';
    const isLevels = effect === 'levels';
    const isColorBalance = effect === 'color balance';
    BLUR_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isBlur));
    HALFTONE_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isHalftone));
    COLOR_FILTER_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isColorFilter));
    SHARPEN_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isSharpen));
    LAPLACIAN_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isLaplacian));
    UNSHARP_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isUnsharp));
    PIXELATE_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isPixelate));
    POSTERIZE_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isPosterize));
    VIGNETTE_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isVignette));
    LEVELS_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isLevels));
    COLOR_BALANCE_WIDGETS.forEach(n => setHidden(findWidget(node, n), !isColorBalance));
    // Widgets that are always hidden regardless of effect:
    //   color_saturation — backing store for the color wheel.
    //   halftone_quality — quality is fixed at 3, kept for back-compat only.
    const alwaysHidden = [
      'color_saturation', 'halftone_quality', 'halftone_dither',
      'vignette_center_x', 'vignette_center_y',
    ].concat(COLOR_BALANCE_HIDDEN_SLIDERS);
    for(const n of alwaysHidden){
      const w = findWidget(node, n);
      if(w){
        // Aggressive hiding so LiteGraph fully ignores the widget for mouse
        // routing AND height accounting:
        //   - `disabled = true` causes processNodeWidgets to `continue` past it
        //     before any hit-test or draw dispatch.
        //   - `type = "hidden"` makes the draw switch fall through to nothing.
        //   - `computeSize → [0, -4]` cancels the per-widget 4-px gap.
        //   - `draw → noop` in case the type switch still routes here.
        //   - `last_y` parked off-canvas so any leaked hit-test never matches.
        w.hidden = true;
        w.disabled = true;
        w.type = 'hidden';
        w.computeSize = () => [0, -4];
        w.draw = () => {};
        w.last_y = -100000;
      }
    }
    // Grow-to-fit only: never shrink the node when widgets change. This keeps
    // the user's manual node size (e.g. enlarged preview area) intact while
    // still ensuring the new widget set has enough room.
    try{
      const min = node.computeSize();
      const cur = node.size || [0, 0];
      const w = Math.max(cur[0] || 0, min[0] || 0);
      const h = Math.max(cur[1] || 0, min[1] || 0);
      if(w !== cur[0] || h !== cur[1]) node.setSize([w, h]);
    }catch(_){ }
    app.graph.setDirtyCanvas(true, true);
  }

  // ---------- preview drawing ----------
  function drawPreview(node, ctx){
    const state = node._imageEffectsState;
    if(!state) return;
    const w = node.size[0];
    const h = node.size[1];
    const pad = 11;

    function widgetsBottomY(n){
      const start = (n.widgets_start_y ?? n.widgetsStartY ?? 0);
      let y = start;
      if(Array.isArray(n.widgets)){
        for(const wg of n.widgets){
          if(!wg) continue;
          let wh = 0;
          try{
            if(typeof wg.computeSize === 'function'){
              const sz = wg.computeSize(w);
              if(Array.isArray(sz)) wh = sz[1] || 0; else if(typeof sz === 'number') wh = sz;
            } else if(typeof wg.height === 'number') {
              wh = wg.height;
            } else { wh = 24; }
          }catch(_){ wh = 24; }
          y += (wh || 0) + 6;
        }
      }
      const minTop = 24;
      return Math.max(y, minTop + 6);
    }

    const safeTop = widgetsBottomY(node) + 18;
    const availableH = Math.max(0, h - safeTop - pad);
    const dh = Math.max(24, availableH);
    const dw = Math.max(0, w - pad*2);
    const x = pad;
    const y = Math.max(safeTop, h - dh - pad);

    ctx.save();
    ctx.fillStyle = "#111";
    ctx.fillRect(x, y, dw, dh);
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.strokeRect(x, y, dw, dh);

    if(!state.outReady){
      ctx.fillStyle = "#bbb";
      ctx.font = "11px sans-serif";
      const msg = state.busy ? "Computing..." : "Run node once to seed preview";
      ctx.fillText(msg, x+12, y+Math.floor(dh/2));
      ctx.restore();
      return;
    }
    const iw = state.outCanvas.width, ih = state.outCanvas.height;
    const scale = Math.min(dw/iw, dh/ih);
    const rw = Math.max(1, Math.floor(iw*scale));
    const rh = Math.max(1, Math.floor(ih*scale));
    const ox = x + Math.floor((dw - rw)/2);
    const oy = y + Math.floor((dh - rh)/2);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(state.outCanvas, 0, 0, iw, ih, ox, oy, rw, rh);
    ctx.restore();
  }

  // ---------- per-node behavior ----------
  function ensureBehavior(node){
    if(node._imageEffectsPreviewAdded) return;
    node._imageEffectsPreviewAdded = true;

    const state = node._imageEffectsState = {
      srcImg: new Image(),
      srcReady: false,
      srcUrl: null,
      srcR: null, srcG: null, srcB: null,
      srcH: null, srcS: null, srcV: null,
      srcL: null, srcAk: null, srcBk: null,
      hsvCached: false, oklabCached: false,
      srcW: 0, srcH_px: 0,
      origW: 0, origH: 0,
      tmpA: null,
      workH: null, workS: null, workV: null,
      workL: null, workAk: null, workBk: null,
      cosBuf: null, sinBuf: null, cosBlur: null, sinBlur: null,
      procR: null, procG: null, procB: null,
      outImageData: null,
      cacheKey: '',
      loadToken: 0,
      outCanvas: document.createElement('canvas'),
      outCtx: null,
      outReady: false,
      busy: false,
      renderTimer: null,
    };
    state.outCtx = state.outCanvas.getContext('2d');

    function revokeUrl(){ try{ if(state.srcUrl){ URL.revokeObjectURL(state.srcUrl); state.srcUrl = null; } }catch(_){} }
    function getWidget(name){ return findWidget(node, name); }
    function getVal(name){ const w = getWidget(name); return w ? w.value : null; }

    function setPreviewId(){
      let w = getWidget('preview_id');
      if(!w && node.addWidget){ w = node.addWidget('string','preview_id', String(node.id ?? '0'), null, {serialize:true}); }
      if(w){
        w.computeSize = () => [0,0]; w.draw = () => {};
        const setVal = (v)=>{ try{ w.value = String(v ?? ''); }catch(_){ };
          if(Array.isArray(node.widgets_values)){
            const idx = node.widgets ? node.widgets.indexOf(w) : -1;
            if(idx>=0) node.widgets_values[idx] = w.value;
          }
        };
        setVal(node.id); setTimeout(()=>setVal(node.id),0);
      }
    }

    function buildPaths(pid, ts){
      const q = `?ts=${ts}`;
      return [
        `/extensions/ComfyUI_Channel_Ops/image_effects_src_${pid}.png${q}`,
        `/extensions/Channel_Ops/image_effects_src_${pid}.png${q}`,
        `/extensions/ChannelOps/image_effects_src_${pid}.png${q}`,
      ];
    }
    async function fetchAsImage(urls, token){
      try{
        for(const u of urls){
          const resp = await fetch(u, {cache:'no-store'});
          if(resp.ok){
            const blob = await resp.blob();
            const obj = URL.createObjectURL(blob);
            revokeUrl();
            state.srcUrl = obj; state.srcImg.src = obj;
            return true;
          }
        }
      }catch(_){ }
      return false;
    }
    function allocBuffers(w, h){
      const n = w * h;
      const f = () => new Float32Array(n);
      state.srcR = f(); state.srcG = f(); state.srcB = f();
      state.srcH = f(); state.srcS = f(); state.srcV = f();
      state.srcL = f(); state.srcAk = f(); state.srcBk = f();
      state.tmpA = f();
      state.workH = f(); state.workS = f(); state.workV = f();
      state.workL = f(); state.workAk = f(); state.workBk = f();
      state.cosBuf = f(); state.sinBuf = f(); state.cosBlur = f(); state.sinBlur = f();
      state.procR = f(); state.procG = f(); state.procB = f();
      state.outCanvas.width = w; state.outCanvas.height = h;
      state.outImageData = state.outCtx.createImageData(w, h);
      state.srcW = w; state.srcH_px = h;
    }
    function decodeSource(){
      const img = state.srcImg;
      const w = img.naturalWidth, h = img.naturalHeight;
      if(w < 1 || h < 1) return;
      const tmp = document.createElement('canvas'); tmp.width = w; tmp.height = h;
      const tctx = tmp.getContext('2d');
      tctx.drawImage(img, 0, 0, w, h);
      const data = tctx.getImageData(0,0,w,h).data;
      const n = w * h;
      if(state.srcW !== w || state.srcH_px !== h) allocBuffers(w, h);
      const sR = state.srcR, sG = state.srcG, sB = state.srcB;
      for(let i=0, p=0; p<n; p++, i+=4){
        sR[p] = data[i]   / 255.0;
        sG[p] = data[i+1] / 255.0;
        sB[p] = data[i+2] / 255.0;
      }
      state.hsvCached = false;
      state.oklabCached = false;
      // Rebuild histograms for the Levels widget's display.
      const hRgb = new Uint32Array(256);
      const hR = new Uint32Array(256);
      const hG = new Uint32Array(256);
      const hB = new Uint32Array(256);
      for(let i=0; i<n; i++){
        const r = Math.min(255, Math.max(0, (state.srcR[i] * 255) | 0));
        const g = Math.min(255, Math.max(0, (state.srcG[i] * 255) | 0));
        const b = Math.min(255, Math.max(0, (state.srcB[i] * 255) | 0));
        const lum = Math.min(255, Math.max(0, ((0.299*r + 0.587*g + 0.114*b) | 0)));
        hR[r]++; hG[g]++; hB[b]++; hRgb[lum]++;
      }
      state.histograms = { rgb: hRgb, red: hR, green: hG, blue: hB };
      state.cacheKey = '';
    }
    function ensureHsv(){
      if(state.hsvCached) return;
      rgbToHsvPlanes(state.srcR, state.srcG, state.srcB, state.srcH, state.srcS, state.srcV);
      state.hsvCached = true;
    }
    function ensureOklab(){
      if(state.oklabCached) return;
      rgbToOklabPlanes(state.srcR, state.srcG, state.srcB, state.srcL, state.srcAk, state.srcBk);
      state.oklabCached = true;
    }
    function loadSrc(tsOverride){
      const wid = getWidget('preview_id');
      const pid = String((wid && wid.value) || node.id || 'A').replace(/[^a-zA-Z0-9_-]/g, "_");
      const ts = (typeof tsOverride==='number' && isFinite(tsOverride)) ? tsOverride : Date.now();
      const paths = buildPaths(pid, ts);
      const token = ++state.loadToken;
      state.srcReady = false;
      state.srcImg.onload = ()=>{
        if(token!==state.loadToken) return;
        try{ decodeSource(); state.srcReady = true; }catch(_){ state.srcReady = false; }
        scheduleRender();
      };
      state.srcImg.onerror = ()=>{ if(token!==state.loadToken) return; state.srcReady = false; scheduleRender(); };
      fetchAsImage(paths, token);
    }
    // Schedule the preview render to fire during browser-idle time. Slider
    // drags generate dense input events; the browser holds back idle
    // callbacks until it has a free moment between them, which keeps the
    // slider's own repaint smooth. The `timeout` ensures the render still
    // ticks at least every 250 ms during continuous interaction, so the
    // preview tries to keep up without ever locking the node. Falls back to
    // setTimeout if requestIdleCallback isn't available.
    state.lastRenderMs = 16;
    const _hasIdle = (typeof window !== 'undefined') &&
                     typeof window.requestIdleCallback === 'function' &&
                     typeof window.cancelIdleCallback === 'function';
    function _cancelPending(){
      if(!state.renderTimer) return;
      if(state.renderTimerType === 'idle'){
        try{ window.cancelIdleCallback(state.renderTimer); }catch(_){}
      } else {
        clearTimeout(state.renderTimer);
      }
      state.renderTimer = null;
    }
    function scheduleRender(){
      _cancelPending();
      const fire = () => {
        state.renderTimer = null;
        const now = (typeof performance !== 'undefined') ? performance.now.bind(performance) : Date.now;
        const t0 = now();
        try{ render(); }catch(e){
          console.error('[ImageEffects] render failed', e);
          state.busy = false;
          app.graph.setDirtyCanvas(true, true);
        }
        state.lastRenderMs = now() - t0;
      };
      if(_hasIdle){
        state.renderTimer = window.requestIdleCallback(fire, { timeout: 25 });
        state.renderTimerType = 'idle';
      } else {
        // Fallback: short timeout. Heavier-than-usual renders still get
        // proportional headroom via lastRenderMs.
        const delay = Math.max(20, Math.min(120, Math.round(state.lastRenderMs * 0.5)));
        state.renderTimer = setTimeout(fire, delay);
        state.renderTimerType = 'timeout';
      }
    }
    function normalizeSource(s){
      s = String(s || 'RGB').trim().toLowerCase();
      switch(s){
        case 'red': return 'r';
        case 'green': return 'g';
        case 'blue': return 'b';
        case 'red+green': return 'r+g';
        case 'red+blue': return 'r+b';
        case 'green+blue': return 'g+b';
        case 'hue': return 'h';
        case 'saturation': return 's';
        case 'value': return 'v';
        case 'rgb': return 'rgb';
        case 'hsv': return 'hsv';
        case 'oklab': return 'oklab';
        default: return 'rgb';
      }
    }
    const RGB_GROUPS = {
      'r':   [true,  false, false],
      'g':   [false, true,  false],
      'b':   [false, false, true ],
      'r+g': [true,  true,  false],
      'r+b': [true,  false, true ],
      'g+b': [false, true,  true ],
      'rgb': [true,  true,  true ],
    };

    function computeProc(effect, radius, source, params){
      const w = state.srcW, h = state.srcH_px;
      const sR = state.srcR, sG = state.srcG, sB = state.srcB;
      const pR = state.procR, pG = state.procG, pB = state.procB;
      const tmp = state.tmpA;

      // Resolve the combined "blur" effect to a specific blur primitive name.
      if(effect === 'blur') effect = resolveBlurEffectName(params.blurMode);

      if(effect === 'halftone'){
        applyHalftone(sR, sG, sB, w, h, params.halftone, tmp, pR, pG, pB);
        return;
      }
      if(effect === 'color filter'){
        applyColorFilter(sR, sG, sB, w, h, params.colorFilter, pR, pG, pB);
        return;
      }
      if(effect === 'sharpen'){
        applySharpen(sR, sG, sB, w, h, params.sharpen, tmp, pR, pG, pB);
        return;
      }
      if(effect === 'laplacian sharpen'){
        applyLaplacianSharpen(sR, sG, sB, w, h, params.laplacian, pR, pG, pB);
        return;
      }
      if(effect === 'unsharp masking'){
        applyUnsharpMask(sR, sG, sB, w, h, params.unsharp, tmp, pR, pG, pB);
        return;
      }
      if(effect === 'pixelate'){
        applyPixelate(sR, sG, sB, w, h, params.pixelate, pR, pG, pB);
        return;
      }
      if(effect === 'posterize'){
        applyPosterize(sR, sG, sB, w, h, params.posterize, pR, pG, pB);
        return;
      }
      if(effect === 'vignette'){
        applyVignette(sR, sG, sB, w, h, params.vignette, pR, pG, pB);
        return;
      }
      if(effect === 'levels'){
        applyLevels(sR, sG, sB, w, h, params.levels, pR, pG, pB);
        return;
      }
      if(effect === 'color balance'){
        applyColorBalance(sR, sG, sB, w, h, params.colorBalance, pR, pG, pB);
        return;
      }
      if(RGB_GROUPS[source]){
        const grp = RGB_GROUPS[source];
        if(grp[0]) applyEffectPlane(effect, sR, tmp, pR, w, h, radius); else pR.set(sR);
        if(grp[1]) applyEffectPlane(effect, sG, tmp, pG, w, h, radius); else pG.set(sG);
        if(grp[2]) applyEffectPlane(effect, sB, tmp, pB, w, h, radius); else pB.set(sB);
        return;
      }
      if(source === 'h' || source === 's' || source === 'v' || source === 'hsv'){
        ensureHsv();
        const wH = state.workH, wS = state.workS, wV = state.workV;
        if(source === 'h'){
          blurHuePlane(state.srcH, tmp, wH, state.cosBuf, state.sinBuf, state.cosBlur, state.sinBlur, w, h, effect, radius);
          wS.set(state.srcS); wV.set(state.srcV);
        } else if(source === 's'){
          wH.set(state.srcH);
          applyEffectPlane(effect, state.srcS, tmp, wS, w, h, radius);
          wV.set(state.srcV);
        } else if(source === 'v'){
          wH.set(state.srcH); wS.set(state.srcS);
          applyEffectPlane(effect, state.srcV, tmp, wV, w, h, radius);
        } else {
          blurHuePlane(state.srcH, tmp, wH, state.cosBuf, state.sinBuf, state.cosBlur, state.sinBlur, w, h, effect, radius);
          applyEffectPlane(effect, state.srcS, tmp, wS, w, h, radius);
          applyEffectPlane(effect, state.srcV, tmp, wV, w, h, radius);
        }
        for(let i=0; i<wS.length; i++){
          let s = wS[i]; if(s<0) s=0; else if(s>1) s=1; wS[i] = s;
          let v = wV[i]; if(v<0) v=0; else if(v>1) v=1; wV[i] = v;
        }
        hsvToRgbPlanes(wH, wS, wV, pR, pG, pB);
        return;
      }
      if(source === 'oklab'){
        ensureOklab();
        applyEffectPlane(effect, state.srcL,  tmp, state.workL,  w, h, radius);
        applyEffectPlane(effect, state.srcAk, tmp, state.workAk, w, h, radius);
        applyEffectPlane(effect, state.srcBk, tmp, state.workBk, w, h, radius);
        oklabToRgbPlanes(state.workL, state.workAk, state.workBk, pR, pG, pB);
        return;
      }
      // fallback: full RGB
      applyEffectPlane(effect, sR, tmp, pR, w, h, radius);
      applyEffectPlane(effect, sG, tmp, pG, w, h, radius);
      applyEffectPlane(effect, sB, tmp, pB, w, h, radius);
    }

    function render(){
      if(!state.srcReady){ state.outReady = false; app.graph.setDirtyCanvas(true,true); return; }
      const effect = String(getVal('effect') || 'Average Blur').toLowerCase();
      const radius = Math.max(0, Math.min(200, parseInt(getVal('radius')) || 0));
      const source = normalizeSource(getVal('source'));
      const mode = String(getVal('blend_mode') || 'Normal').toLowerCase();
      const w = state.srcW, h = state.srcH_px;
      if(w < 1 || h < 1){ state.outReady = false; return; }

      // Scale radius to preview resolution so the preview matches actual output.
      const origMax = Math.max(state.origW|0, state.origH|0);
      const previewMax = Math.max(w, h);
      const scale = origMax > 0 ? (previewMax / origMax) : 1.0;
      const effR = Math.max(0, Math.round(radius * scale));

      const halftoneParams = {
        shape: getVal('halftone_shape') || 'Dot',
        inverse: !!getVal('halftone_inverse'),
        size: Math.max(1, Math.round((parseFloat(getVal('halftone_size')) || 18) * scale)),
        angle: parseFloat(getVal('halftone_angle')) || 0,
        contrast: parseInt(getVal('halftone_contrast')) || 0,
        brightness: parseInt(getVal('halftone_brightness')) || 0,
        quality: Math.max(1, Math.min(5, parseInt(getVal('halftone_quality')) || 2)),
        // Dithering is always on — Bayer ordered dither scaled by size.
        dither: 100,
      };
      const colorFilterParams = {
        hue: parseFloat(getVal('color_hue')) || 0,
        saturation: parseFloat(getVal('color_saturation')) || 0,
        density: parseFloat(getVal('color_density')) || 0,
        preserve: parseFloat(getVal('color_preserve_highlights')) || 0,
      };
      // For the new spatial effects, also scale "size-like" parameters by
      // (previewMax / origMax) so the preview matches the full-res output.
      const sharpenParams = {
        amount: parseInt(getVal('sharpen_amount')) || 0,
        radius: Math.max(1, Math.round((parseInt(getVal('sharpen_radius')) || 1) * scale)),
        threshold: parseInt(getVal('sharpen_threshold')) || 0,
      };
      const laplacianParams = {
        amount: parseFloat(getVal('laplacian_amount')) || 0,
        kernel: getVal('laplacian_kernel') || '3x3 (8-neighbor)',
      };
      const unsharpParams = {
        amount: parseFloat(getVal('usm_amount')) || 0,
        radius: Math.max(1, Math.round((parseInt(getVal('usm_radius')) || 3) * scale)),
        threshold: parseInt(getVal('usm_threshold')) || 0,
      };
      const pixelateParams = {
        size: Math.max(1, Math.round((parseInt(getVal('pixelate_size')) || 16) * scale)),
        mode: getVal('pixelate_mode') || 'Pixelate',
      };
      const posterizeParams = {
        levels: parseInt(getVal('posterize_levels')) || 4,
        mode: getVal('posterize_mode') || 'RGB',
        dither: Math.max(0, Math.min(100, parseInt(getVal('posterize_dither')) || 0)),
        ditherMode: getVal('posterize_dither_mode') || 'None',
      };
      const vignetteParams = {
        amount: parseInt(getVal('vignette_amount')) || 0,
        size: parseInt(getVal('vignette_size')) || 0,
        feather: parseInt(getVal('vignette_feather')) || 0,
        roundness: parseInt(getVal('vignette_roundness')) || 0,
        cx: Math.max(0, Math.min(100, parseInt(getVal('vignette_center_x')) || 50)),
        cy: Math.max(0, Math.min(100, parseInt(getVal('vignette_center_y')) || 50)),
      };
      const blurMode = String(getVal('blur_mode') || 'Gaussian');
      const levelsParams = {
        channel: getVal('levels_channel') || 'RGB',
        inBlack: parseInt(getVal('levels_in_black')) || 0,
        inWhite: parseInt(getVal('levels_in_white')) || 255,
        gamma: parseFloat(getVal('levels_gamma')) || 1.0,
        outBlack: parseInt(getVal('levels_out_black')) || 0,
        outWhite: parseInt(getVal('levels_out_white')) || 255,
      };
      const colorBalanceParams = {
        sh_r: parseInt(getVal('cb_shadow_red'))    || 0,
        sh_g: parseInt(getVal('cb_shadow_green'))  || 0,
        sh_b: parseInt(getVal('cb_shadow_blue'))   || 0,
        mid_r: parseInt(getVal('cb_midtone_red'))    || 0,
        mid_g: parseInt(getVal('cb_midtone_green'))  || 0,
        mid_b: parseInt(getVal('cb_midtone_blue'))   || 0,
        hi_r: parseInt(getVal('cb_highlight_red'))   || 0,
        hi_g: parseInt(getVal('cb_highlight_green')) || 0,
        hi_b: parseInt(getVal('cb_highlight_blue'))  || 0,
      };

      let key;
      if(effect === 'halftone'){
        const p = halftoneParams;
        key = 'halftone|' + p.shape + '|' + p.inverse + '|' + p.size + '|' + p.angle + '|' + p.contrast + '|' + p.brightness + '|' + p.quality + '|' + p.dither;
      } else if(effect === 'color filter'){
        const p = colorFilterParams;
        key = 'color_filter|' + p.hue + '|' + p.saturation + '|' + p.density + '|' + p.preserve;
      } else if(effect === 'sharpen'){
        const p = sharpenParams;
        key = 'sharpen|' + p.amount + '|' + p.radius + '|' + p.threshold;
      } else if(effect === 'laplacian sharpen'){
        const p = laplacianParams;
        key = 'laplacian|' + p.amount + '|' + p.kernel;
      } else if(effect === 'unsharp masking'){
        const p = unsharpParams;
        key = 'unsharp|' + p.amount + '|' + p.radius + '|' + p.threshold;
      } else if(effect === 'pixelate'){
        const p = pixelateParams;
        key = 'pixelate|' + p.size + '|' + p.mode;
      } else if(effect === 'posterize'){
        const p = posterizeParams;
        key = 'posterize|' + p.levels + '|' + p.mode + '|' + p.dither + '|' + p.ditherMode;
      } else if(effect === 'vignette'){
        const p = vignetteParams;
        key = 'vignette|' + p.amount + '|' + p.size + '|' + p.feather + '|' + p.roundness + '|' + p.cx + '|' + p.cy;
      } else if(effect === 'levels'){
        const p = levelsParams;
        key = 'levels|' + p.channel + '|' + p.inBlack + '|' + p.inWhite + '|' + p.gamma + '|' + p.outBlack + '|' + p.outWhite;
      } else if(effect === 'color balance'){
        const p = colorBalanceParams;
        key = 'cb|' + p.sh_r + '|' + p.sh_g + '|' + p.sh_b + '|' + p.mid_r + '|' + p.mid_g + '|' + p.mid_b + '|' + p.hi_r + '|' + p.hi_g + '|' + p.hi_b;
      } else {
        // Blur (including legacy effect names). Key includes blur_mode.
        key = effect + '|' + blurMode + '|' + effR + '|' + source;
      }
      if(key !== state.cacheKey){
        state.busy = true;
        app.graph.setDirtyCanvas(true,true);
        computeProc(effect, effR, source, {
          halftone: halftoneParams,
          colorFilter: colorFilterParams,
          sharpen: sharpenParams,
          laplacian: laplacianParams,
          unsharp: unsharpParams,
          pixelate: pixelateParams,
          posterize: posterizeParams,
          vignette: vignetteParams,
          levels: levelsParams,
          colorBalance: colorBalanceParams,
          blurMode: blurMode,
        });
        state.cacheKey = key;
        state.busy = false;
      }

      const sR = state.srcR, sG = state.srcG, sB = state.srcB;
      const pR = state.procR, pG = state.procG, pB = state.procB;
      const od = state.outImageData.data;
      const N = w * h;
      for(let p=0, i=0; p<N; p++, i+=4){
        const ar = sR[p], ag = sG[p], ab = sB[p];
        const cr = pR[p], cg = pG[p], cb = pB[p];
        let rr, gg, bb2;
        if(mode === 'darker color'){
          const sa = ar+ag+ab, sb = cr+cg+cb;
          if(sb < sa){ rr=cr; gg=cg; bb2=cb; } else { rr=ar; gg=ag; bb2=ab; }
        } else if(mode === 'lighter color'){
          const sa = ar+ag+ab, sb = cr+cg+cb;
          if(sb > sa){ rr=cr; gg=cg; bb2=cb; } else { rr=ar; gg=ag; bb2=ab; }
        } else {
          rr = blendPixel(ar, cr, mode);
          gg = blendPixel(ag, cg, mode);
          bb2 = blendPixel(ab, cb, mode);
        }
        if(rr<0)rr=0; else if(rr>1)rr=1;
        if(gg<0)gg=0; else if(gg>1)gg=1;
        if(bb2<0)bb2=0; else if(bb2>1)bb2=1;
        od[i]   = (rr*255)|0;
        od[i+1] = (gg*255)|0;
        od[i+2] = (bb2*255)|0;
        od[i+3] = 255;
      }
      state.outCtx.putImageData(state.outImageData, 0, 0);
      state.outReady = true;
      app.graph.setDirtyCanvas(true,true);
    }

    function bindChanges(){
      if(!node.widgets) return;
      node.widgets.forEach(w => {
        const orig = w.callback || w.onChange;
        const isEffect = w && w.name === 'effect';
        const cb = function(){
          if(orig) try{ orig.apply(this, arguments); }catch(_e){}
          if(isEffect) try{ syncWidgets(node); }catch(_e){}
          scheduleRender();
        };
        w.callback = cb; w.onChange = cb;
      });
    }

    function applyTooltips(){
      const wEff = getWidget('effect');
      const wRad = getWidget('radius');
      const wSrc = getWidget('source');
      const wMode = getWidget('blend_mode');
      if(wEff) wEff.tooltip = wEff.description = [
        'Average Blur: flat box average inside the radius.',
        'Gaussian Blur: bell-weighted smooth blur.',
        'Average Edge Blur: averages just the perimeter ring at the radius (Bresenham 8-way circle).',
        'Halftone: dotted/lined pattern with brightness driving cell size.',
        'Color Filter: tints the image with a chosen hue.',
        'Sharpen: unsharp-mask; negative amount = soften.',
        'Laplacian Sharpen: edge boost via the Laplacian operator (4- or 8-neighbor).',
        'Unsharp Masking: classic Gaussian unsharp mask with amount/radius/threshold.',
        'Pixelate / Mosaic: square blocks of average color (Mosaic adds grout lines).',
        'Posterize: quantises each channel to N levels.',
        'Vignette: radial dark/light falloff toward the corners.',
        'Levels: input black/gamma/white + output black/white mapping.',
        'Color Balance: per-tonal-range (shadows/midtones/highlights) RGB shifts.',
      ].join('\n');
      if(wRad) wRad.tooltip = wRad.description = 'Effect radius (0-200). 0 disables the effect.';
      if(wSrc) wSrc.tooltip = wSrc.description = [
        'Channel(s) the effect runs on.',
        'R/G/B: single RGB channel.',
        'R+G / R+B / G+B: a pair of RGB channels.',
        'H/S/V: single HSV channel (Hue averaged on the unit circle).',
        'RGB / HSV / Oklab: all three channels of that space.',
      ].join('\n');
      if(wMode) wMode.tooltip = wMode.description = 'Blend mode used to composite the effect result over the original.';
      const halftoneTips = {
        halftone_shape: 'Halftone cell shape.',
        halftone_inverse: 'Invert the halftone mask (dots become holes).',
        halftone_size: 'Halftone cell size in pixels.',
        halftone_angle: 'Rotation of the halftone grid, degrees.',
        halftone_contrast: 'Pre-contrast applied to source before halftoning (-100..100).',
        halftone_brightness: 'Pre-brightness offset applied to source before halftoning (-100..100).',
        halftone_quality: 'Subpixel samples per pixel (1-5). Higher = smoother edges, slower.',
        halftone_dither: 'Ordered (Bayer 8x8) dithering strength. 0 = smooth halftone, 100 = crisp B/W with dithered tonal transitions (Floyd-Steinberg-style look).',
      };
      for(const [n, t] of Object.entries(halftoneTips)){
        const w = getWidget(n);
        if(w){ w.tooltip = w.description = t; }
      }
      const colorTips = {
        color_hue: 'Filter hue in degrees (0-360). Synced with the color wheel.',
        color_density: 'Strength of the tint (0-255). Wheel cursor distance from center.',
        color_preserve_highlights: 'How much bright pixels resist the tint (0-100). 100 = highlights untouched, 0 = all pixels tinted equally.',
      };
      for(const [n, t] of Object.entries(colorTips)){
        const w = getWidget(n);
        if(w){ w.tooltip = w.description = t; }
      }
      // Angle wheel shares the halftone_angle tooltip semantics.
      const wAngleWheel = getWidget('halftone_angle_wheel');
      if(wAngleWheel) wAngleWheel.tooltip = wAngleWheel.description = 'Drag the line to set the halftone angle. Mirrors the angle slider.';

      const extraTips = {
        sharpen_amount: 'Sharpen strength (-200 to 200). Positive sharpens, negative softens.',
        sharpen_radius: 'Detail radius for the unsharp mask.',
        sharpen_threshold: 'Minimum local contrast for sharpening (suppresses noise).',
        laplacian_amount: 'Laplacian sharpen strength (0-5). Higher = crisper edges, more haloing.',
        laplacian_kernel: '4-neighbor = softer (edges only). 8-neighbor = stronger (includes diagonals).',
        usm_amount: 'Unsharp mask strength as a multiplier (1.0 = 100%).',
        usm_radius: 'Gaussian blur radius of the unsharp mask — larger = coarser detail enhanced.',
        usm_threshold: 'Minimum local contrast (0-255) before sharpening is applied (suppresses noise).',
        pixelate_size: 'Block size in pixels.',
        pixelate_mode: 'Pixelate = solid blocks. Mosaic = blocks with dark grout lines.',
        posterize_levels: 'Levels per channel (2-32). Lower = more bands.',
        posterize_dither_mode: [
          'Dithering algorithm used before quantising:',
          'None — hard quantization (visible banding on gradients).',
          'Bayer — ordered 8x8 dither, fast and stable.',
          'Random — white-noise offset, grainy look.',
          'Floyd-Steinberg — classic error diffusion, highest quality (slow).',
          'Atkinson — variant of FS used by early Macs (slow, slightly lighter).',
        ].join('\n'),
        vignette_amount: 'Vignette strength. Positive darkens, negative lightens.',
        vignette_size: 'Inner radius (where the vignette starts).',
        vignette_feather: 'Falloff width — softness of the vignette edge.',
        vignette_roundness: 'Negative = wider (landscape), 0 = circular, positive = taller (portrait).',
        levels_channel: 'Which channel(s) the levels operation applies to.',
        levels_in_black: 'Input black point — pixels at or below this become black.',
        levels_in_white: 'Input white point — pixels at or above this become white.',
        levels_gamma: 'Midtone correction. >1 brightens midtones, <1 darkens them.',
        levels_out_black: 'Output black point — lifts the blacks.',
        levels_out_white: 'Output white point — lowers the whites.',
        cb_shadow_red: 'Shadow region: shift toward red (+) or cyan (-).',
        cb_shadow_green: 'Shadow region: shift toward green (+) or magenta (-).',
        cb_shadow_blue: 'Shadow region: shift toward blue (+) or yellow (-).',
        cb_midtone_red: 'Midtone region: shift toward red (+) or cyan (-).',
        cb_midtone_green: 'Midtone region: shift toward green (+) or magenta (-).',
        cb_midtone_blue: 'Midtone region: shift toward blue (+) or yellow (-).',
        cb_highlight_red: 'Highlight region: shift toward red (+) or cyan (-).',
        cb_highlight_green: 'Highlight region: shift toward green (+) or magenta (-).',
        cb_highlight_blue: 'Highlight region: shift toward blue (+) or yellow (-).',
      };
      for(const [n, t] of Object.entries(extraTips)){
        const w = getWidget(n);
        if(w){ w.tooltip = w.description = t; }
      }
      const wLevels = getWidget('levels_widget');
      if(wLevels) wLevels.tooltip = wLevels.description = 'Histogram + draggable handles. Drag B (black), gamma, W (white) on top bar to set input range; drag B/W on bottom bar to set output range.';
    }

    // Helper: write a value into a node widget and fire its callback. This
    // mirrors how a user dragging the standard slider would behave — the
    // bound onChange runs (which calls scheduleRender) and the canvas dirties.
    function setNodeWidgetValue(n, name, value){
      const w = n.widgets ? n.widgets.find(x => x && x.name === name) : null;
      if(!w) return;
      w.value = value;
      if(typeof w.callback === 'function'){ try{ w.callback(value); }catch(_){ } }
    }

    // Convert a browser pointer event to node-local coords using LGraphCanvas's
    // own transform helper so pan/zoom are taken into account correctly.
    function eventToNodeLocal(e, n){
      const canvas = app.canvas;
      if(canvas && typeof canvas.convertEventToCanvasOffset === 'function'){
        const off = canvas.convertEventToCanvasOffset(e);
        return [off[0] - n.pos[0], off[1] - n.pos[1]];
      }
      return [0, 0];
    }

    // ---------- color wheel widget ----------
    // Convention: 0deg = top (red), increasing clockwise. Distance from center
    // maps to density (0..255). Saturation is fixed at 1 (the disc visualises
    // a full-sat ring at the rim; pulling toward center weakens the effect
    // rather than desaturating it).
    function addColorWheelWidget(){
      if(findWidget(node, 'color_wheel')) return;
      const wheel = {
        type: 'custom',
        name: 'color_wheel',
        value: 0,
        serialize: false,
        hidden: true,
        _keepOwnComputeSize: true,
        _isCircularWheel: true,
        last_y: 0,
        options: {},
        computeSize(width){
          if(this.hidden) return [0, -4];
          const s = Math.max(80, Math.min(width - 20, 200));
          return [width, s + 6];
        },
        draw(ctx, n, widget_width, y, H){
          if(this.hidden) return;
          this.last_y = y;
          const sz = Math.max(80, Math.min(widget_width - 20, 200));
          const px = Math.floor((widget_width - sz) / 2);
          const py = y;
          ctx.save();
          ctx.drawImage(getWheelCanvas(), px, py, sz, sz);
          const hueWid = findWidget(n, 'color_hue');
          const densWid = findWidget(n, 'color_density');
          const hue = hueWid ? Number(hueWid.value) || 0 : 0;
          const density = densWid ? Math.min(255, Math.max(0, Number(densWid.value) || 0)) : 0;
          const rMax = sz / 2 - 2;
          // hue=0 at top, clockwise: cursor_dx = sin(h), cursor_dy = -cos(h)
          const ang = (hue % 360) * Math.PI / 180;
          const r = (density / 255) * rMax;
          const cxp = px + sz / 2 + Math.sin(ang) * r;
          const cyp = py + sz / 2 + (-Math.cos(ang)) * r;
          ctx.lineWidth = 2; ctx.strokeStyle = '#000';
          ctx.beginPath(); ctx.arc(cxp, cyp, 6, 0, Math.PI * 2); ctx.stroke();
          ctx.lineWidth = 1; ctx.strokeStyle = '#fff';
          ctx.beginPath(); ctx.arc(cxp, cyp, 6, 0, Math.PI * 2); ctx.stroke();
          ctx.restore();
        },
        // Geometry for the drag-capture path below. Computes Y from scratch
        // each call so we don't depend on last_y being current (e.g. when the
        // wheel was just made visible by an effect change).
        _geom(n){
          const width = n.size[0];
          const sz = Math.max(80, Math.min(width - 20, 200));
          const px = Math.floor((width - sz) / 2);
          // Prefer last_y set during the most recent draw — it is the exact Y
          // LiteGraph used to render the widget. Fall back to a computed Y if
          // the widget has never been drawn (e.g. click before first frame).
          const py = (typeof this.last_y === 'number' && this.last_y > 0)
            ? this.last_y : computeWidgetY(n, this);
          return { cx: px + sz / 2, cy: py + sz / 2, rMax: sz / 2 - 2 };
        },
        mouse(event, pos, n){ return wheelMouseHandler(this, event, pos, n); },
        _onPick(n, dx, dy, geom){
          const dist = Math.sqrt(dx*dx + dy*dy);
          const d = Math.min(geom.rMax, dist);
          const density = Math.round((d / Math.max(1, geom.rMax)) * 255);
          // atan2(dx, -dy): 0 at up, +pi/2 at right (clockwise). Matches the
          // standard color-wheel convention used to render the disc above.
          let hue = Math.atan2(dx, -dy) * 180 / Math.PI;
          if(hue < 0) hue += 360;
          setNodeWidgetValue(n, 'color_hue', Math.round(hue));
          setNodeWidgetValue(n, 'color_density', density);
        },
      };
      let insertIdx = node.widgets.length;
      const after = findWidget(node, 'color_preserve_highlights');
      if(after){
        const i = node.widgets.indexOf(after);
        if(i >= 0) insertIdx = i + 1;
      }
      node.widgets.splice(insertIdx, 0, wheel);
    }

    // ---------- halftone angle wheel widget ----------
    // Simple circle with a line from center pointing in the current angle.
    // Convention: 0deg = right (3 o'clock), positive angle = CCW (math),
    // which matches the existing halftone_angle slider semantics.
    function addHalftoneAngleWheelWidget(){
      if(findWidget(node, 'halftone_angle_wheel')) return;
      const wheel = {
        type: 'custom',
        name: 'halftone_angle_wheel',
        value: 0,
        serialize: false,
        hidden: true,
        _keepOwnComputeSize: true,
        _isCircularWheel: true,
        last_y: 0,
        options: {},
        computeSize(width){
          if(this.hidden) return [0, -4];
          const s = 80;
          return [width, s + 6];
        },
        draw(ctx, n, widget_width, y, H){
          if(this.hidden) return;
          this.last_y = y;
          const sz = 80;
          const px = Math.floor((widget_width - sz) / 2);
          const py = y;
          const cx = px + sz / 2, cy = py + sz / 2;
          const r = sz / 2 - 4;
          ctx.save();
          // Disc
          ctx.fillStyle = '#dadada';
          ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 1;
          ctx.stroke();
          // Line indicator. Convention matched to the drag math: positive
          // angle = CW from 3 o'clock (so dragging the tip CW makes the angle
          // increase and the pattern rotate CW too).
          const angWid = findWidget(n, 'halftone_angle');
          const angDeg = angWid ? (Number(angWid.value) || 0) : 0;
          const ang = angDeg * Math.PI / 180;
          const tipX = cx + Math.cos(ang) * (r - 2);
          const tipY = cy + Math.sin(ang) * (r - 2);
          ctx.strokeStyle = '#222';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(tipX, tipY);
          ctx.stroke();
          ctx.fillStyle = '#222';
          ctx.beginPath(); ctx.arc(cx, cy, 2, 0, Math.PI * 2); ctx.fill();
          ctx.restore();
        },
        _geom(n){
          const width = n.size[0];
          const sz = 80;
          const px = Math.floor((width - sz) / 2);
          const py = (typeof this.last_y === 'number' && this.last_y > 0)
            ? this.last_y : computeWidgetY(n, this);
          return { cx: px + sz / 2, cy: py + sz / 2, rMax: sz / 2 - 4 };
        },
        mouse(event, pos, n){ return wheelMouseHandler(this, event, pos, n); },
        _onPick(n, dx, dy, geom){
          if(dx === 0 && dy === 0) return;
          // Drag direction is inverted relative to the pattern's rotation
          // sense, so we flip the sign. atan2(dy, dx) gives the canvas angle
          // (Y-down, CW positive) which now matches the halftone pattern's
          // rotation direction visually.
          let angDeg = Math.atan2(dy, dx) * 180 / Math.PI;
          angDeg = Math.round(angDeg * 100) / 100;
          setNodeWidgetValue(n, 'halftone_angle', angDeg);
        },
      };
      let insertIdx = node.widgets.length;
      const after = findWidget(node, 'halftone_angle');
      if(after){
        const i = node.widgets.indexOf(after);
        if(i >= 0) insertIdx = i + 1;
      }
      node.widgets.splice(insertIdx, 0, wheel);
    }

    // ---------- levels custom widget ----------
    // Histogram + draggable in/out level handles. Backed by the standard
    // levels_in_black/_in_white/_gamma/_out_black/_out_white sliders.
    function addLevelsWidget(){
      if(findWidget(node, 'levels_widget')) return;
      const LAYOUT = { padding: 10, histH: 64, barH: 8, handleH: 12, gap: 5 };
      const totalH = LAYOUT.histH + LAYOUT.gap + LAYOUT.barH + LAYOUT.handleH +
                     LAYOUT.gap + LAYOUT.barH + LAYOUT.handleH + 4;

      function geom(n){
        const width = n.size[0];
        const x = LAYOUT.padding;
        const w = Math.max(40, width - 2 * LAYOUT.padding);
        const y = (typeof widget.last_y === 'number' && widget.last_y > 0)
          ? widget.last_y : computeWidgetY(n, widget);
        return { x, y, w, h: totalH };
      }
      function inputBarY(g){ return g.y + LAYOUT.histH + LAYOUT.gap; }
      function inputHandleCenterY(g){ return inputBarY(g) + LAYOUT.barH + LAYOUT.handleH / 2; }
      function outputBarY(g){ return inputBarY(g) + LAYOUT.barH + LAYOUT.handleH + LAYOUT.gap; }
      function outputHandleCenterY(g){ return outputBarY(g) + LAYOUT.barH + LAYOUT.handleH / 2; }

      function curHandlePositions(n, g){
        const inB = Number(findWidget(n, 'levels_in_black') ?.value || 0);
        const inW = Number(findWidget(n, 'levels_in_white') ?.value || 255);
        const gam = Number(findWidget(n, 'levels_gamma') ?.value || 1);
        const outB = Number(findWidget(n, 'levels_out_black') ?.value || 0);
        const outW = Number(findWidget(n, 'levels_out_white') ?.value || 255);
        const xOfV = (v) => g.x + (v / 255) * g.w;
        const inBlackX = xOfV(inB);
        const inWhiteX = xOfV(inW);
        // gamma position: gamma=1 at midpoint; 10 at left; 0.1 at right.
        const gRel = 0.5 - Math.log10(Math.max(0.01, Math.min(99.99, gam))) / 2;
        const gammaX = inBlackX + Math.max(0, Math.min(1, gRel)) * (inWhiteX - inBlackX);
        const outBlackX = xOfV(outB);
        const outWhiteX = xOfV(outW);
        return { inB, inW, gam, outB, outW, inBlackX, inWhiteX, gammaX, outBlackX, outWhiteX };
      }

      function drawHandle(ctx, x, y, fill, stroke){
        ctx.beginPath();
        ctx.moveTo(x, y - 6);
        ctx.lineTo(x - 5, y + 4);
        ctx.lineTo(x + 5, y + 4);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      const widget = {
        type: 'custom',
        name: 'levels_widget',
        value: 0,
        serialize: false,
        hidden: true,
        _keepOwnComputeSize: true,
        _isLevelsWidget: true,
        _activeHandle: null,
        options: {},
        computeSize(width){
          if(this.hidden) return [0, -4];
          return [width, totalH];
        },
        draw(ctx, n, widget_width, y, H){
          if(this.hidden) return;
          this.last_y = y;
          const g = { x: LAYOUT.padding, y, w: Math.max(40, widget_width - 2 * LAYOUT.padding), h: totalH };
          const channel = String((findWidget(n, 'levels_channel') || {}).value || 'RGB');
          const histKey = channel.toLowerCase();
          const state = n._imageEffectsState;
          const hist = (state && state.histograms) ? state.histograms[histKey] : null;

          ctx.save();
          // Histogram background
          ctx.fillStyle = '#1a1a1a';
          ctx.fillRect(g.x, g.y, g.w, LAYOUT.histH);
          ctx.strokeStyle = 'rgba(255,255,255,0.12)';
          ctx.strokeRect(g.x, g.y, g.w, LAYOUT.histH);
          if(hist){
            let maxV = 0;
            for(let i=0; i<256; i++){ if(hist[i] > maxV) maxV = hist[i]; }
            if(maxV > 0){
              ctx.fillStyle = channel === 'Red' ? 'rgba(220,80,80,0.9)' :
                              channel === 'Green' ? 'rgba(80,220,80,0.9)' :
                              channel === 'Blue' ? 'rgba(80,140,255,0.9)' :
                              'rgba(210,210,210,0.9)';
              const barW = g.w / 256;
              for(let i=0; i<256; i++){
                const bh = Math.round((hist[i] / maxV) * LAYOUT.histH);
                if(bh <= 0) continue;
                ctx.fillRect(g.x + i * barW, g.y + LAYOUT.histH - bh, Math.max(1, barW), bh);
              }
            }
          }

          // Input gradient bar
          const ibY = inputBarY(g);
          const gradIn = ctx.createLinearGradient(g.x, ibY, g.x + g.w, ibY);
          gradIn.addColorStop(0, '#000'); gradIn.addColorStop(1, '#fff');
          ctx.fillStyle = gradIn;
          ctx.fillRect(g.x, ibY, g.w, LAYOUT.barH);

          const pos = curHandlePositions(n, g);
          const inHy = inputHandleCenterY(g);
          drawHandle(ctx, pos.inBlackX, inHy, '#000', '#fff');
          drawHandle(ctx, pos.gammaX,   inHy, '#808080', '#fff');
          drawHandle(ctx, pos.inWhiteX, inHy, '#fff', '#000');

          // Output gradient bar
          const obY = outputBarY(g);
          const gradOut = ctx.createLinearGradient(g.x, obY, g.x + g.w, obY);
          gradOut.addColorStop(0, '#000'); gradOut.addColorStop(1, '#fff');
          ctx.fillStyle = gradOut;
          ctx.fillRect(g.x, obY, g.w, LAYOUT.barH);
          const outHy = outputHandleCenterY(g);
          drawHandle(ctx, pos.outBlackX, outHy, '#000', '#fff');
          drawHandle(ctx, pos.outWhiteX, outHy, '#fff', '#000');
          ctx.restore();
        },
        _hitTest(n, pointerPos){
          if(this.hidden) return null;
          const g = geom(n);
          if(pointerPos[0] < g.x - 6 || pointerPos[0] > g.x + g.w + 6) return null;
          if(pointerPos[1] < g.y || pointerPos[1] > g.y + g.h) return null;
          const pos = curHandlePositions(n, g);
          const inHy = inputHandleCenterY(g);
          const outHy = outputHandleCenterY(g);
          const tol = 10;
          // Output handles only if pointer is in lower half of widget
          if(Math.abs(pointerPos[1] - outHy) < 12){
            const dB = Math.abs(pointerPos[0] - pos.outBlackX);
            const dW = Math.abs(pointerPos[0] - pos.outWhiteX);
            if(Math.min(dB, dW) < tol){
              return dB < dW ? 'out_black' : 'out_white';
            }
          }
          // Input handles
          if(Math.abs(pointerPos[1] - inHy) < 12 || (pointerPos[1] > g.y && pointerPos[1] < inHy + 6)){
            const dB = Math.abs(pointerPos[0] - pos.inBlackX);
            const dG = Math.abs(pointerPos[0] - pos.gammaX);
            const dW = Math.abs(pointerPos[0] - pos.inWhiteX);
            const min = Math.min(dB, dG, dW);
            if(min < tol){
              if(min === dB) return 'in_black';
              if(min === dW) return 'in_white';
              return 'in_gamma';
            }
          }
          return null;
        },
        _applyDrag(n, pointerPos){
          const g = geom(n);
          const xNorm = Math.max(0, Math.min(1, (pointerPos[0] - g.x) / Math.max(1, g.w)));
          const value = Math.round(xNorm * 255);
          const inB = Number(findWidget(n, 'levels_in_black') ?.value || 0);
          const inW = Number(findWidget(n, 'levels_in_white') ?.value || 255);
          const outB = Number(findWidget(n, 'levels_out_black') ?.value || 0);
          const outW = Number(findWidget(n, 'levels_out_white') ?.value || 255);

          if(this._activeHandle === 'in_black'){
            const v = Math.max(0, Math.min(inW - 1, value));
            setNodeWidgetValue(n, 'levels_in_black', v);
          } else if(this._activeHandle === 'in_white'){
            const v = Math.max(inB + 1, Math.min(255, value));
            setNodeWidgetValue(n, 'levels_in_white', v);
          } else if(this._activeHandle === 'in_gamma'){
            // Position within [in_black, in_white] -> gamma 10..0.1
            const range = Math.max(1, inW - inB);
            const rel = Math.max(0.01, Math.min(0.99, (value - inB) / range));
            const gamma = Math.pow(10, (0.5 - rel) * 2);
            setNodeWidgetValue(n, 'levels_gamma', Math.round(gamma * 100) / 100);
          } else if(this._activeHandle === 'out_black'){
            const v = Math.max(0, Math.min(outW - 1, value));
            setNodeWidgetValue(n, 'levels_out_black', v);
          } else if(this._activeHandle === 'out_white'){
            const v = Math.max(outB + 1, Math.min(255, value));
            setNodeWidgetValue(n, 'levels_out_white', v);
          }
        },
        mouse(event, pos, n){
          if(this.hidden) return false;
          const t = event && event.type;
          if(t !== 'pointerdown' && t !== 'mousedown') return false;
          const handle = this._hitTest(n, pos);
          if(!handle) return false;
          this._activeHandle = handle;
          startGenericDrag(n, this, pos);
          return true;
        },
      };
      let insertIdx = node.widgets.length;
      const after = findWidget(node, 'levels_channel');
      if(after){
        const i = node.widgets.indexOf(after);
        if(i >= 0) insertIdx = i + 1;
      }
      node.widgets.splice(insertIdx, 0, widget);
    }

    // Generic drag for non-circular widgets (used by Levels). The widget must
    // implement _applyDrag(node, pointerPos). Uses document listeners for the
    // actual move/up dispatch — same robustness as the wheel drag.
    function startGenericDrag(n, widget, initialPos){
      if(widget._dragActive) return;
      widget._dragActive = true;
      try{ widget._applyDrag(n, initialPos); }catch(err){ console.error(err); }
      scheduleRender();
      app.graph.setDirtyCanvas(true, true);
      const onMove = (mv) => {
        if(!widget._dragActive) return;
        try{
          const loc = eventToNodeLocal(mv, n);
          widget._applyDrag(n, loc);
          scheduleRender();
          app.graph.setDirtyCanvas(true, true);
        }catch(err){ console.error(err); }
        mv.preventDefault(); mv.stopPropagation();
      };
      const onUp = (up) => {
        widget._dragActive = false;
        widget._activeHandle = null;
        document.removeEventListener('pointermove', onMove, true);
        document.removeEventListener('pointerup', onUp, true);
        document.removeEventListener('pointercancel', onUp, true);
        up.preventDefault(); up.stopPropagation();
      };
      document.addEventListener('pointermove', onMove, true);
      document.addEventListener('pointerup', onUp, true);
      document.addEventListener('pointercancel', onUp, true);
    }

    // ---------- shared wheel-drag mechanic ----------
    // Both wheels (color filter + halftone angle) share this drag flow. Either
    // entry point — LiteGraph's widget.mouse dispatch OR the node.onMouseDown
    // override — calls startWheelDrag. The wheel._dragActive flag prevents
    // double-firing if both paths run for the same click.
    function startWheelDrag(n, wheel, dx, dy, geom){
      if(wheel._dragActive) return;
      wheel._dragActive = true;
      try{ wheel._onPick(n, dx, dy, geom); }catch(err){ console.error(err); }
      scheduleRender();
      app.graph.setDirtyCanvas(true, true);
      const onMove = (mv) => {
        if(!wheel._dragActive) return;
        try{
          const loc = eventToNodeLocal(mv, n);
          const g = (typeof wheel._geom === 'function') ? wheel._geom(n) : geom;
          wheel._onPick(n, loc[0] - g.cx, loc[1] - g.cy, g);
          scheduleRender();
          app.graph.setDirtyCanvas(true, true);
        }catch(err){ console.error(err); }
        mv.preventDefault(); mv.stopPropagation();
      };
      const onUp = (up) => {
        wheel._dragActive = false;
        document.removeEventListener('pointermove', onMove, true);
        document.removeEventListener('pointerup', onUp, true);
        document.removeEventListener('pointercancel', onUp, true);
        up.preventDefault(); up.stopPropagation();
      };
      document.addEventListener('pointermove', onMove, true);
      document.addEventListener('pointerup', onUp, true);
      document.addEventListener('pointercancel', onUp, true);
    }

    // LiteGraph widget-level mouse path. Used when the click is dispatched
    // via processNodeWidgets (some LiteGraph versions); returning true on
    // pointerdown captures the widget so subsequent moves/up route here too.
    // We still set up document-level listeners in startWheelDrag because in
    // other versions LiteGraph stops dispatching widget.mouse mid-drag.
    function wheelMouseHandler(wheel, event, pos, n){
      if(wheel.hidden) return false;
      const t = event && event.type;
      if(t !== 'pointerdown' && t !== 'mousedown') return false;
      const geom = (typeof wheel._geom === 'function') ? wheel._geom(n) : null;
      if(!geom) return false;
      const dx = pos[0] - geom.cx;
      const dy = pos[1] - geom.cy;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if(dist > geom.rMax + 8) return false;
      startWheelDrag(n, wheel, dx, dy, geom);
      return true;
    }

    // ---------- vignette center widget ----------
    // Box showing the source image with a draggable crosshair marker for
    // vignette center. Backed by vignette_center_x / _y sliders (0-100).
    function addVignetteWidget(){
      if(findWidget(node, 'vignette_widget')) return;
      const HEIGHT = 130;
      const PAD = 10;

      function imageBox(node, widget){
        const width = node.size[0];
        const y = (typeof widget.last_y === 'number' && widget.last_y > 0)
          ? widget.last_y : computeWidgetY(node, widget);
        const boxX = PAD;
        const boxY = y + 4;
        const boxW = Math.max(40, width - 2 * PAD);
        const boxH = HEIGHT - 8;
        const state = node._imageEffectsState;
        const iw = state ? state.srcW : 0;
        const ih = state ? state.srcH_px : 0;
        let drawW = boxW, drawH = boxH;
        if(iw > 0 && ih > 0){
          const ia = iw / ih;
          const ba = boxW / boxH;
          if(ia > ba){ drawH = boxW / ia; } else { drawW = boxH * ia; }
        }
        const drawX = boxX + Math.floor((boxW - drawW) / 2);
        const drawY = boxY + Math.floor((boxH - drawH) / 2);
        return { boxX, boxY, boxW, boxH, drawX, drawY, drawW, drawH };
      }

      const widget = {
        type: 'custom',
        name: 'vignette_widget',
        value: 0,
        serialize: false,
        hidden: true,
        _keepOwnComputeSize: true,
        _isPositionalWidget: true,
        last_y: 0,
        options: {},
        computeSize(width){
          if(this.hidden) return [0, -4];
          return [width, HEIGHT];
        },
        // LiteGraph routes pointerdown to widgets whose type is 'custom' AND
        // whose .mouse() returns truthy. Capture here as a primary path; the
        // node.onMouseDown override is still installed as a fallback.
        mouse(event, pos, n){
          if(this.hidden) return false;
          const t = event && event.type;
          if(t !== 'pointerdown' && t !== 'mousedown') return false;
          const hit = this._hitTest(n, pos);
          if(!hit) return false;
          startGenericDrag(n, this, pos);
          return true;
        },
        draw(ctx, n, widget_width, y, H){
          if(this.hidden) return;
          this.last_y = y;
          const g = imageBox(n, this);
          ctx.save();
          // Grid background.
          ctx.fillStyle = '#1a1a1a';
          ctx.fillRect(g.boxX, g.boxY, g.boxW, g.boxH);
          ctx.strokeStyle = 'rgba(255,255,255,0.08)';
          ctx.lineWidth = 1;
          // 25%, 50%, 75% gridlines inside the inner image box.
          for(let i = 1; i < 4; i++){
            const fx = g.drawX + (g.drawW * i) / 4;
            ctx.beginPath(); ctx.moveTo(fx, g.drawY); ctx.lineTo(fx, g.drawY + g.drawH); ctx.stroke();
            const fy = g.drawY + (g.drawH * i) / 4;
            ctx.beginPath(); ctx.moveTo(g.drawX, fy); ctx.lineTo(g.drawX + g.drawW, fy); ctx.stroke();
          }
          // Image-area outline.
          ctx.strokeStyle = 'rgba(255,255,255,0.30)';
          ctx.lineWidth = 1;
          ctx.strokeRect(g.drawX, g.drawY, g.drawW, g.drawH);
          // Outer widget bounds.
          ctx.strokeStyle = 'rgba(255,255,255,0.12)';
          ctx.strokeRect(g.boxX, g.boxY, g.boxW, g.boxH);

          // Crosshair at current center.
          const cx_n = Math.max(0, Math.min(100, Number(findWidget(n, 'vignette_center_x')?.value || 50))) / 100;
          const cy_n = Math.max(0, Math.min(100, Number(findWidget(n, 'vignette_center_y')?.value || 50))) / 100;
          const cx = g.drawX + cx_n * g.drawW;
          const cy = g.drawY + cy_n * g.drawH;
          // Outline ring
          ctx.lineWidth = 2; ctx.strokeStyle = '#000';
          ctx.beginPath(); ctx.arc(cx, cy, 9, 0, Math.PI * 2); ctx.stroke();
          ctx.lineWidth = 1; ctx.strokeStyle = '#fff';
          ctx.beginPath(); ctx.arc(cx, cy, 9, 0, Math.PI * 2); ctx.stroke();
          // Crosshair lines
          ctx.beginPath();
          ctx.moveTo(cx - 14, cy); ctx.lineTo(cx - 5, cy);
          ctx.moveTo(cx + 5, cy);  ctx.lineTo(cx + 14, cy);
          ctx.moveTo(cx, cy - 14); ctx.lineTo(cx, cy - 5);
          ctx.moveTo(cx, cy + 5);  ctx.lineTo(cx, cy + 14);
          ctx.strokeStyle = '#000'; ctx.lineWidth = 2; ctx.stroke();
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
          ctx.restore();
        },
        _hitTest(n, pos){
          if(this.hidden) return null;
          const g = imageBox(n, this);
          if(pos[0] < g.boxX || pos[0] > g.boxX + g.boxW) return null;
          if(pos[1] < g.boxY || pos[1] > g.boxY + g.boxH) return null;
          return 'center';
        },
        _applyDrag(n, pos){
          const g = imageBox(n, this);
          if(g.drawW <= 0 || g.drawH <= 0) return;
          let nx = (pos[0] - g.drawX) / g.drawW;
          let ny = (pos[1] - g.drawY) / g.drawH;
          if(nx < 0) nx = 0; else if(nx > 1) nx = 1;
          if(ny < 0) ny = 0; else if(ny > 1) ny = 1;
          setNodeWidgetValue(n, 'vignette_center_x', Math.round(nx * 100));
          setNodeWidgetValue(n, 'vignette_center_y', Math.round(ny * 100));
        },
      };

      let insertIdx = node.widgets.length;
      const after = findWidget(node, 'vignette_roundness');
      if(after){
        const i = node.widgets.indexOf(after);
        if(i >= 0) insertIdx = i + 1;
      }
      node.widgets.splice(insertIdx, 0, widget);
    }

    // ---------- color balance triple-wheel widget ----------
    // Three small color discs (Shadows / Midtones / Highlights). Hue = tint
    // direction, distance from center = intensity. Backed by the 9 hidden
    // cb_<range>_<rgb> sliders so the values still serialise.
    function addColorBalanceWidget(){
      if(findWidget(node, 'cb_widget')) return;
      const PAD = 10;
      const LABEL_H = 14;
      const HEIGHT = 100;

      function layout(node, widget){
        const width = node.size[0];
        const y = (typeof widget.last_y === 'number' && widget.last_y > 0)
          ? widget.last_y : computeWidgetY(node, widget);
        const w = Math.max(120, width - 2 * PAD);
        // 3 wheels horizontally, gap of 8px.
        const gap = 8;
        const wheelSize = Math.max(50, Math.min(76, Math.floor((w - 2 * gap) / 3)));
        const totalW = wheelSize * 3 + gap * 2;
        const startX = PAD + Math.floor((w - totalW) / 2);
        const wheelY = y + LABEL_H + 2;
        return { y, w, wheelSize, gap, startX, wheelY };
      }

      function wheelCenter(L, i){
        return {
          cx: L.startX + i * (L.wheelSize + L.gap) + L.wheelSize / 2,
          cy: L.wheelY + L.wheelSize / 2,
          rMax: L.wheelSize / 2 - 2,
        };
      }

      const RANGES = [
        { label: 'Shadows',    prefix: 'cb_shadow_' },
        { label: 'Midtones',   prefix: 'cb_midtone_' },
        { label: 'Highlights', prefix: 'cb_highlight_' },
      ];

      function shiftsToPos(r, g, b){
        const tr = 0.5 + r / 200;
        const tg = 0.5 + g / 200;
        const tb = 0.5 + b / 200;
        const cr = Math.max(0, Math.min(1, tr));
        const cg = Math.max(0, Math.min(1, tg));
        const cb_ = Math.max(0, Math.min(1, tb));
        // Inline RGB→HSV (matches getWheelCanvas convention: hue 0..1).
        const mx = Math.max(cr, cg, cb_);
        const mn = Math.min(cr, cg, cb_);
        const d = mx - mn;
        let hue = 0;
        if(d > 1e-8){
          if(mx === cr) hue = (((cg - cb_) / d) / 6) % 1;
          else if(mx === cg) hue = ((2 + (cb_ - cr) / d) / 6) % 1;
          else hue = ((4 + (cr - cg) / d) / 6) % 1;
          if(hue < 0) hue += 1;
        }
        const s = mx > 0 ? d / mx : 0;
        // Distance for cursor placement = saturation * value (so equal shifts
        // give equal cursor distances).
        return { hue: hue, distance: s * mx };
      }

      function setRangeFromWheel(n, prefix, hue, distance){
        // hue 0..1, distance 0..1. Convert to RGB shifts in -100..100.
        // Inline HSV→RGB at full sat/value, then subtract 0.5 and scale.
        const h6 = hue * 6;
        const ii = Math.floor(h6);
        const f = h6 - ii;
        const p = 0; // sat=1, value=1: p = v*(1-s) = 0
        const q = 1 - f;
        const t = f;
        let r, g, b;
        switch(((ii % 6) + 6) % 6){
          case 0: r=1; g=t; b=p; break;
          case 1: r=q; g=1; b=p; break;
          case 2: r=p; g=1; b=t; break;
          case 3: r=p; g=q; b=1; break;
          case 4: r=t; g=p; b=1; break;
          default: r=1; g=p; b=q;
        }
        const sr = Math.round((r - 0.5) * 2 * 100 * distance);
        const sg = Math.round((g - 0.5) * 2 * 100 * distance);
        const sb = Math.round((b - 0.5) * 2 * 100 * distance);
        setNodeWidgetValue(n, prefix + 'red',   Math.max(-100, Math.min(100, sr)));
        setNodeWidgetValue(n, prefix + 'green', Math.max(-100, Math.min(100, sg)));
        setNodeWidgetValue(n, prefix + 'blue',  Math.max(-100, Math.min(100, sb)));
      }

      const widget = {
        type: 'custom',
        name: 'cb_widget',
        value: 0,
        serialize: false,
        hidden: true,
        _keepOwnComputeSize: true,
        _isPositionalWidget: true,
        last_y: 0,
        _activeRangeIdx: -1,
        options: {},
        computeSize(width){
          if(this.hidden) return [0, -4];
          return [width, HEIGHT];
        },
        mouse(event, pos, n){
          if(this.hidden) return false;
          const t = event && event.type;
          if(t !== 'pointerdown' && t !== 'mousedown') return false;
          const hit = this._hitTest(n, pos);
          if(!hit) return false;
          startGenericDrag(n, this, pos);
          return true;
        },
        draw(ctx, n, widget_width, y, H){
          if(this.hidden) return;
          this.last_y = y;
          const L = layout(n, this);
          ctx.save();
          for(let i=0; i<3; i++){
            const c = wheelCenter(L, i);
            // Label
            ctx.fillStyle = '#cfcfcf';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(RANGES[i].label, c.cx, y + LABEL_H - 2);
            // Disc
            const disc = getWheelCanvas();
            const sz = L.wheelSize;
            ctx.drawImage(disc, c.cx - sz/2, c.cy - sz/2, sz, sz);
            // Cursor from current shifts
            const r = parseInt(findWidget(n, RANGES[i].prefix + 'red')?.value)   || 0;
            const g = parseInt(findWidget(n, RANGES[i].prefix + 'green')?.value) || 0;
            const b = parseInt(findWidget(n, RANGES[i].prefix + 'blue')?.value)  || 0;
            const pos = shiftsToPos(r, g, b);
            // Color-wheel disc convention: hue=0 at top, clockwise (sin/-cos).
            const ang = pos.hue * Math.PI * 2;
            const rr = pos.distance * c.rMax;
            const cxp = c.cx + Math.sin(ang) * rr;
            const cyp = c.cy - Math.cos(ang) * rr;
            ctx.lineWidth = 2; ctx.strokeStyle = '#000';
            ctx.beginPath(); ctx.arc(cxp, cyp, 4, 0, Math.PI * 2); ctx.stroke();
            ctx.lineWidth = 1; ctx.strokeStyle = '#fff';
            ctx.beginPath(); ctx.arc(cxp, cyp, 4, 0, Math.PI * 2); ctx.stroke();
          }
          ctx.restore();
        },
        _hitTest(n, pos){
          if(this.hidden) return null;
          const L = layout(n, this);
          for(let i=0; i<3; i++){
            const c = wheelCenter(L, i);
            const dx = pos[0] - c.cx;
            const dy = pos[1] - c.cy;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if(dist <= c.rMax + 8){
              this._activeRangeIdx = i;
              return 'range_' + i;
            }
          }
          return null;
        },
        _applyDrag(n, pos){
          const i = this._activeRangeIdx;
          if(i < 0 || i > 2) return;
          const L = layout(n, this);
          const c = wheelCenter(L, i);
          const dx = pos[0] - c.cx;
          const dy = pos[1] - c.cy;
          const dist = Math.min(c.rMax, Math.sqrt(dx*dx + dy*dy));
          const distance = c.rMax > 0 ? dist / c.rMax : 0;
          // Convert (dx, -dy) to hue with 0=top, clockwise. atan2(dx, -dy)
          // matches the disc's color layout (see getWheelCanvas).
          let hue = Math.atan2(dx, -dy) / (Math.PI * 2);
          if(hue < 0) hue += 1;
          setRangeFromWheel(n, RANGES[i].prefix, hue, distance);
        },
      };

      let insertIdx = node.widgets.length;
      // Insert before the first hidden cb slider so it stays at the top of
      // the color-balance widget group.
      const after = findWidget(node, 'cb_shadow_red');
      if(after){
        const i = node.widgets.indexOf(after);
        if(i >= 0) insertIdx = i;
      }
      node.widgets.splice(insertIdx, 0, widget);
    }

    // ---------- node-level pointer capture ----------
    // Overrides node.onMouseDown to intercept wheel clicks BEFORE LiteGraph's
    // node-drag logic activates. Per-instance, so multiple Filter Ops nodes
    // on the same graph stay isolated.
    function setupWheelCapture(){
      const origDown = node.onMouseDown;
      node.onMouseDown = function(e, pos, gc){
        if(node.widgets){
          // Circular wheels (color wheel, halftone angle wheel).
          for(const w of node.widgets){
            if(!w || !w._isCircularWheel || w.hidden) continue;
            if(w._dragActive) continue;
            const geom = (typeof w._geom === 'function') ? w._geom(node) : null;
            if(!geom) continue;
            const dx = pos[0] - geom.cx;
            const dy = pos[1] - geom.cy;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if(dist > geom.rMax + 8) continue;
            startWheelDrag(node, w, dx, dy, geom);
            return true;
          }
          // Levels widget (multi-handle, rectangular).
          for(const w of node.widgets){
            if(!w || !w._isLevelsWidget || w.hidden) continue;
            if(w._dragActive) continue;
            const handle = (typeof w._hitTest === 'function') ? w._hitTest(node, pos) : null;
            if(!handle) continue;
            w._activeHandle = handle;
            startGenericDrag(node, w, pos);
            return true;
          }
          // Other positional widgets (vignette center, color-balance wheels).
          for(const w of node.widgets){
            if(!w || !w._isPositionalWidget || w.hidden) continue;
            if(w._dragActive) continue;
            const hit = (typeof w._hitTest === 'function') ? w._hitTest(node, pos) : null;
            if(!hit) continue;
            startGenericDrag(node, w, pos);
            return true;
          }
        }
        return origDown ? origDown.apply(this, arguments) : false;
      };
    }

    // ---------- top-of-node reorder + separator ----------
    // Move `effect` and `blend_mode` to the very top of the widget list and
    // insert a thin separator beneath them, so those two always stay together
    // regardless of which effect is active.
    function reorderTopWidgets(){
      if(!node.widgets) return;
      const ws = node.widgets;
      const byName = (n) => ws.find(w => w && w.name === n);
      const effect = byName('effect');
      const blend = byName('blend_mode');
      if(!effect || !blend) return;
      let sep = byName('_top_separator');
      if(!sep){
        sep = {
          type: 'custom',
          name: '_top_separator',
          value: 0,
          serialize: false,
          _keepOwnComputeSize: true,
          hidden: false,
          options: {},
          computeSize(width){ return [width, 10]; },
          draw(ctx, n, widget_width, y, H){
            ctx.save();
            ctx.strokeStyle = 'rgba(255,255,255,0.18)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(10, y + 5);
            ctx.lineTo(widget_width - 10, y + 5);
            ctx.stroke();
            ctx.restore();
          },
        };
      }
      // Already in desired order? Skip to avoid churn on every onConfigure.
      if(ws[0] === effect && ws[1] === blend && ws[2] === sep) return;
      const others = ws.filter(w => w !== effect && w !== blend && w !== sep);
      node.widgets = [effect, blend, sep, ...others];
    }

    // Helper exposed for the module-level onConfigure handler.
    node._imageEffectsReorderTop = reorderTopWidgets;

    // Replace underscores with spaces in the visible widget label. The
    // underlying widget.name (used for serialization and kwargs) stays the
    // same — only the displayed label changes.
    function prettifyLabels(){
      if(!node.widgets) return;
      for(const w of node.widgets){
        if(!w || !w.name) continue;
        if(w.name.startsWith('_') || w.name === 'preview_id') continue;
        if(w.label && w.label !== w.name) continue;
        if(w.name.indexOf('_') < 0) continue;
        w.label = w.name.replace(/_/g, ' ');
      }
    }

    setPreviewId();
    addColorWheelWidget();
    addHalftoneAngleWheelWidget();
    addLevelsWidget();
    addVignetteWidget();
    addColorBalanceWidget();
    reorderTopWidgets();
    setupWheelCapture();
    bindChanges();
    applyTooltips();
    prettifyLabels();
    syncWidgets(node);
    loadSrc();

    node._imageEffectsLoad = loadSrc;
    node._imageEffectsSync = () => syncWidgets(node);
    const prevExec = node.onExecuted;
    node.onExecuted = function(){ if(prevExec) try{ prevExec.apply(this, arguments); }catch(_e){} try{ loadSrc(); }catch(_e){} };
  }

  app.registerExtension({
    name: EXT_NAME,
    async setup(){
      try{
        api.addEventListener('image_effects_preview', (ev)=>{
          const detail = ev?.detail ?? ev;
          const pid = String(detail?.preview_id || '');
          const ts = (typeof detail?.ts === 'number') ? Math.floor(detail.ts*1000) : Date.now();
          const origW = (typeof detail?.orig_w === 'number') ? detail.orig_w : 0;
          const origH = (typeof detail?.orig_h === 'number') ? detail.orig_h : 0;
          if(!pid) return;
          const nodes = (app?.graph?._nodes || []).filter(n => (n.comfyClass || n.type || '').toString().includes('ImageEffectsNode'));
          for(const n of nodes){
            try{
              if(!n._imageEffectsPreviewAdded) ensureBehavior(n);
              let w = n.widgets ? n.widgets.find(w => w && (w.name==='preview_id' || w.label==='preview_id')) : null;
              const widVal = w ? String(w.value || '') : '';
              const nodeIdOk = (typeof n.id === 'number' && n.id >= 0) ? String(n.id) : '';
              const matches = (widVal && widVal===pid) || (nodeIdOk && nodeIdOk===pid);
              if(matches){
                const st = n._imageEffectsState;
                if(st){
                  st.origW = origW || st.origW || 0;
                  st.origH = origH || st.origH || 0;
                  st.cacheKey = '';
                }
                if(n._imageEffectsLoad) n._imageEffectsLoad(ts);
              }
            }catch(_e){}
          }
        });
      }catch(_e){}
    },
    async beforeRegisterNodeDef(nodeType, nodeData){
      const nm = (nodeData?.name || nodeType?.name || '').toString();
      if(nm.includes('ImageEffectsNode')){
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(){
          if(origCreated) try{ origCreated.apply(this, arguments); }catch(_e){}
          try{ ensureBehavior(this); }catch(_e){}
        };
        // Re-sync widget visibility after a workflow has finished loading.
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info){
          if(origConfigure) try{ origConfigure.apply(this, arguments); }catch(_e){}
          const self = this;
          // Defer to next tick so widgets_values have been applied to widget.value first.
          setTimeout(()=>{
            try{
              // Re-apply the top-of-node reorder (separator etc.) in case the
              // saved order placed widgets differently.
              if(self._imageEffectsReorderTop) self._imageEffectsReorderTop();
              // Workflows saved before the reorder used positional order
              // [effect, radius, source, blend_mode, ...]. Detect the old
              // layout by checking whether blend_mode now holds a value that
              // is NOT a valid blend mode string (typically a number from the
              // radius slot), and remap the first 4 widgets by name.
              if(info && Array.isArray(info.widgets_values)){
                const widgets = self.widgets || [];
                const byName = (n) => widgets.find(w => w && w.name === n);
                const blendW = byName('blend_mode');
                if(blendW){
                  const looksWrong = (typeof blendW.value !== 'string') || !BLEND_MODE_NAMES.has(blendW.value);
                  if(looksWrong){
                    const v = info.widgets_values;
                    const effectW = byName('effect');
                    const radiusW = byName('radius');
                    const sourceW = byName('source');
                    if(effectW && v.length > 0) effectW.value = v[0];
                    if(radiusW && v.length > 1) radiusW.value = v[1];
                    if(sourceW && v.length > 2) sourceW.value = v[2];
                    if(v.length > 3) blendW.value = v[3];
                  }
                }
                // Migrate old blur effect names → new "Blur" + blur_mode. The
                // raw saved value at position 0 is the effect string from
                // before the combine.
                const rawEffect = String(info.widgets_values[0] || '');
                const blurMap = {
                  'Average Blur': 'Average',
                  'Average Edge Blur': 'Edge Average',
                  'Gaussian Blur': 'Gaussian',
                };
                if(blurMap[rawEffect]){
                  const effectW = byName('effect');
                  const modeW = byName('blur_mode');
                  if(effectW) effectW.value = 'Blur';
                  if(modeW) modeW.value = blurMap[rawEffect];
                }
              }
            }catch(_e){}
            try{ if(self._imageEffectsSync) self._imageEffectsSync(); }catch(_e){}
          }, 0);
        };
        const origDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx){
          if(origDraw) try{ origDraw.apply(this, arguments); }catch(_e){}
          try{ if(this._imageEffectsPreviewAdded) drawPreview(this, ctx); }catch(_e){}
        };
      }
    },
  });
})();
