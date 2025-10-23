// ChannelOps - In-node live preview (canvas)
// This script runs in the ComfyUI frontend and augments the ChannelOpsNode UI.
// It displays a canvas below the node's sliders and updates in real-time as you tweak values.
// It uses a per-node preview image saved by the backend to web/channel_ops_preview_<preview_id>.png on each run.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const EXT_NAME = "Channel_Ops.Preview";

  // Draw helper using node-scoped state (called from prototype wrapper)
  function drawPreview(node, ctx){
    const state = node._channelOpsState;
    if(!state) return;
    const w = node.size[0];
    const h = node.size[1];
    const pad = 11;

    // Estimate where widgets end to avoid overlap
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
            } else {
              wh = 24;
            }
          }catch(_){ wh = 24; }
          y += (wh || 0) + 6; // include a slightly larger spacing
        }
      }
      // Also ensure we are at least below the title bar
      const minTop = 24; // typical title height
      return Math.max(y, minTop + 6);
    }

    const safeTop = widgetsBottomY(node) + 12; // extra padding below widgets
    const availableH = Math.max(0, h - safeTop - pad);
    // Fill the remaining height to make the preview scale with node size
    const desiredH = Math.max(24, availableH);
    const dh = desiredH;
    const dw = Math.max(0, w - pad*2);
    const x = pad;
    const y = Math.max(safeTop, h - dh - pad);

    ctx.save();
    // Background panel
    ctx.fillStyle = "#111";
    ctx.fillRect(x, y, dw, dh);
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.strokeRect(x, y, dw, dh);

    if(availableH < 16){
      // Too small to draw anything safely
      ctx.fillStyle = "#bbb";
      ctx.font = "11px sans-serif";
      const msg = "Enlarge node to show preview";
      ctx.fillText(msg, x+8, Math.max(pad+12, y+12));
      ctx.restore();
      return;
    }

    if(state.baseLoaded && state.offscreen.width>0 && state.offscreen.height>0){
      const iw = state.offscreen.width;
      const ih = state.offscreen.height;
      const scale = Math.min(dw/iw, dh/ih);
      const rw = Math.max(1, Math.floor(iw*scale));
      const rh = Math.max(1, Math.floor(ih*scale));
      const ox = x + Math.floor((dw - rw)/2);
      const oy = y + Math.floor((dh - rh)/2);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(state.offscreen, 0, 0, iw, ih, ox, oy, rw, rh);
    } else {
      ctx.fillStyle = "#bbb";
      ctx.font = "11px sans-serif";
      const msg = "Run node once to seed preview";
      ctx.fillText(msg, x+12, y+Math.floor(dh/2));
    }
    ctx.restore();
  }

  function ensureBehavior(node){
    if(node._channelOpsPreviewAdded) return;
    node._channelOpsPreviewAdded = true;

    // Internal state for offscreen rendering
    const state = node._channelOpsState = {
      baseImg: new Image(),
      baseLoaded: false,
      offscreen: document.createElement('canvas'),
      offctx: null,
      previewHeight: 256,
      loadToken: 0,
      pendingRetry: null,
      objectUrl: null,
      // Secondary source image for overwrite-from-image
      srcImg: new Image(),
      srcLoaded: false,
      srcOffscreen: document.createElement('canvas'),
      srcOffctx: null,
      srcPendingRetry: null,
      srcLoadToken: 0,
      srcObjectUrl: null,
    };
    state.offctx = state.offscreen.getContext('2d');
    state.srcOffctx = state.srcOffscreen.getContext('2d');

    // Do not alter node height; draw within current area to avoid layout issues

    function revokeObjectUrl(){
      if(state.objectUrl){
        try{ URL.revokeObjectURL(state.objectUrl); }catch(_){ }
        state.objectUrl = null;
      }
    }

    function revokeSrcObjectUrl(){
      if(state.srcObjectUrl){
        try{ URL.revokeObjectURL(state.srcObjectUrl); }catch(_){ }
        state.srcObjectUrl = null;
      }
    }

    function resetPreview(){
      // Clear current preview and show placeholder until a new image loads
      state.baseLoaded = false;
      revokeObjectUrl();
      try { state.baseImg.src = ""; } catch(_) {}
      try {
        state.offscreen.width = 0; state.offscreen.height = 0;
      } catch(_) {}
      // Clear secondary source
      state.srcLoaded = false;
      revokeSrcObjectUrl();
      try { state.srcImg.src = ""; } catch(_) {}
      try { state.srcOffscreen.width = 0; state.srcOffscreen.height = 0; } catch(_) {}
      app.graph.setDirtyCanvas(true,true);
    }

  async function loadBase(tsOverride){
      // Single-attempt load (used by retry scheduler)
      state.baseLoaded = false;
      const wid = getWidget('preview_id');
      const raw = wid ? (wid.value ?? '') : '';
      const nodeIdOk = (typeof node.id === 'number' && node.id >= 0) ? String(node.id) : '';
      const userPid = String(raw || '').replace(/[^a-zA-Z0-9_-]/g, "_");
      // Strictly use per-node identifiers to avoid cross-node preview bleed; no global 'A' fallback
      const candidates = Array.from(new Set([
        userPid || undefined,
        nodeIdOk || undefined
      ].filter(Boolean)));
  const ts = (typeof tsOverride === 'number' && isFinite(tsOverride)) ? tsOverride : Date.now();
      let ci = 0;
      const token = ++state.loadToken;
      // Reset src to avoid cached decode pipelines on same Image object
      try { state.baseImg.src = ""; } catch(_) {}
      state.baseImg.crossOrigin = "anonymous";
      state.baseImg.onload = () => {
        if(token !== state.loadToken) return; // stale
        state.baseLoaded = true;
        renderEffect();
        app.graph.setDirtyCanvas(true,true);
      };
      state.baseImg.onerror = () => {
        if(token !== state.loadToken) return; // stale
        // This error handler is a last resort; normally we set the src from a fetched blob
        state.baseLoaded = false;
        app.graph.setDirtyCanvas(true,true);
      };

      // Try candidates by fetching as blob with no-store, then set object URL
      while(ci < candidates.length){
        if(token !== state.loadToken) return;
        // Prefer current project path ComfyUI_Channel_Ops, then Channel_Ops, then legacy ChannelOps (backward compatibility)
        const tryPaths = [
          `/extensions/ComfyUI_Channel_Ops/channel_ops_preview_${candidates[ci]}.png?ts=${ts}`,
          `/extensions/Channel_Ops/channel_ops_preview_${candidates[ci]}.png?ts=${ts}`,
          `/extensions/ChannelOps/channel_ops_preview_${candidates[ci]}.png?ts=${ts}`
        ];
        try{
          let blob = null;
          for(const p of tryPaths){
            const resp = await fetch(p, { cache: 'no-store' });
            if(resp.ok){ blob = await resp.blob(); break; }
          }
          if(!blob){ ci++; continue; }
          if(token !== state.loadToken) return;
          revokeObjectUrl();
          const obj = URL.createObjectURL(blob);
          state.objectUrl = obj;
          state.baseImg.src = obj;
          return;
        }catch(_e){
          ci++;
        }
      }
      // All candidates failed
      state.baseLoaded = false;
      app.graph.setDirtyCanvas(true,true);
    }

    // Retry loader to cope with FS write latency after execution
  function loadBaseWithRetry(tsHint){
      // Cancel any pending retry chain
      if(state.pendingRetry){
        state.pendingRetry.cancelled = true;
        state.pendingRetry = null;
      }
      // Immediately clear any stale preview so we don't show old images
      resetPreview();
      const schedule = [100, 250, 500, 1000, 1500]; // ms
      const token = ++state.loadToken;
      let idx = -1;

      const task = { cancelled: false };
      state.pendingRetry = task;

      const attempt = () => {
        if(task.cancelled) return;
        // Attempt a load; on error we'll schedule next delay
        let settled = false;
        const prevOnLoad = state.baseImg.onload;
        const prevOnError = state.baseImg.onerror;

  state.baseImg.onload = () => {
          if(task.cancelled || token !== state.loadToken) return;
          settled = true;
          // restore handlers to default flow
          state.baseImg.onload = prevOnLoad;
          state.baseImg.onerror = prevOnError;
          state.baseLoaded = true;
          renderEffect();
          app.graph.setDirtyCanvas(true,true);
          state.pendingRetry = null;
        };
        state.baseImg.onerror = () => {
          if(task.cancelled || token !== state.loadToken) return;
          // restore handlers and schedule next try
          state.baseImg.onload = prevOnLoad;
          state.baseImg.onerror = prevOnError;
          idx += 1;
          if(idx < schedule.length){
            setTimeout(() => {
              if(task.cancelled || token !== state.loadToken) return;
              loadBase(tsHint); // reattempt with provided ts if any
              attempt();
            }, schedule[idx]);
          } else {
            // Give up; keep last state
            state.pendingRetry = null;
            app.graph.setDirtyCanvas(true,true);
          }
        };

        // Kick the first attempt
        loadBase(tsHint);
      };

      attempt();
    }

    async function loadSrc(tsOverride){
      // Attempt to load per-node secondary source preview used in Overwrite from Image
      state.srcLoaded = false;
      const wid = getWidget('preview_id');
      const raw = wid ? (wid.value ?? '') : '';
      const nodeIdOk = (typeof node.id === 'number' && node.id >= 0) ? String(node.id) : '';
      const userPid = String(raw || '').replace(/[^a-zA-Z0-9_-]/g, "_");
      const candidates = Array.from(new Set([
        userPid || undefined,
        nodeIdOk || undefined
      ].filter(Boolean)));
      const ts = (typeof tsOverride === 'number' && isFinite(tsOverride)) ? tsOverride : Date.now();
      let ci = 0;
      const token = ++state.srcLoadToken;
      try { state.srcImg.src = ""; } catch(_) {}
      state.srcImg.crossOrigin = "anonymous";
      state.srcImg.onload = () => {
        if(token !== state.srcLoadToken) return; // stale
        state.srcLoaded = true;
        renderEffect();
        app.graph.setDirtyCanvas(true,true);
      };
      state.srcImg.onerror = () => {
        if(token !== state.srcLoadToken) return; // stale
        state.srcLoaded = false;
        app.graph.setDirtyCanvas(true,true);
      };
      while(ci < candidates.length){
        if(token !== state.srcLoadToken) return;
        const tryPaths = [
          `/extensions/ComfyUI_Channel_Ops/channel_ops_preview_src_${candidates[ci]}.png?ts=${ts}`,
          `/extensions/Channel_Ops/channel_ops_preview_src_${candidates[ci]}.png?ts=${ts}`,
          `/extensions/ChannelOps/channel_ops_preview_src_${candidates[ci]}.png?ts=${ts}`
        ];
        try{
          let blob = null;
          for(const p of tryPaths){
            const resp = await fetch(p, { cache: 'no-store' });
            if(resp.ok){ blob = await resp.blob(); break; }
          }
          if(!blob){ ci++; continue; }
          if(token !== state.srcLoadToken) return;
          revokeSrcObjectUrl();
          const obj = URL.createObjectURL(blob);
          state.srcObjectUrl = obj;
          state.srcImg.src = obj;
          return;
        }catch(_e){ ci++; }
      }
      state.srcLoaded = false;
      app.graph.setDirtyCanvas(true,true);
    }

    function loadSrcWithRetry(tsHint){
      if(state.srcPendingRetry){ state.srcPendingRetry.cancelled = true; state.srcPendingRetry = null; }
      const schedule = [100, 250, 500, 1000, 1500];
      const token = ++state.srcLoadToken;
      let idx = -1;
      const task = { cancelled: false };
      state.srcPendingRetry = task;
      const attempt = () => {
        if(task.cancelled) return;
        const prevOnLoad = state.srcImg.onload;
        const prevOnError = state.srcImg.onerror;
        state.srcImg.onload = () => {
          if(task.cancelled || token !== state.srcLoadToken) return;
          state.srcImg.onload = prevOnLoad; state.srcImg.onerror = prevOnError;
          state.srcLoaded = true;
          renderEffect();
          app.graph.setDirtyCanvas(true,true);
          state.srcPendingRetry = null;
        };
        state.srcImg.onerror = () => {
          if(task.cancelled || token !== state.srcLoadToken) return;
          state.srcImg.onload = prevOnLoad; state.srcImg.onerror = prevOnError;
          idx += 1;
          if(idx < schedule.length){
            setTimeout(() => { if(task.cancelled || token !== state.srcLoadToken) return; loadSrc(tsHint); attempt(); }, schedule[idx]);
          } else { state.srcPendingRetry = null; app.graph.setDirtyCanvas(true,true); }
        };
        loadSrc(tsHint);
      };
      attempt();
    }

    function getWidget(name){
      if(!node.widgets) return null;
      return node.widgets.find(w => (w && w.name === name) || (w && w.label === name));
    }

    function getValue(name){
      const w = getWidget(name);
      if(!w) return null;
      return w.value;
    }

    function renderEffect(){
      if(!state.baseLoaded) return;
      const op = (getValue('operation') || '').toString().toLowerCase().replace(/\s+/g,'_');
      const src = (getValue('Source') || '').toString();
      const dst = (getValue('Destination') || '').toString();
      const amount255 = parseFloat(getValue('amount') || 0) || 0;
      // If this operation depends on a secondary image, ensure it's loading
      if(op === 'overwrite_from_image' && !state.srcLoaded){
        try{
          if(node._channelOpsLoadSrcWithRetry) node._channelOpsLoadSrcWithRetry();
          else if(node._channelOpsLoadSrc) node._channelOpsLoadSrc();
        }catch(_e){}
      }

      // Prepare offscreen canvas based on base image
      const w = state.baseImg.naturalWidth;
      const h = state.baseImg.naturalHeight;
      state.offscreen.width = w;
      state.offscreen.height = h;
      const ctx = state.offctx;
      ctx.drawImage(state.baseImg, 0, 0);
      const imgData = ctx.getImageData(0, 0, w, h);
      const data = imgData.data; // RGBA 0..255
      // Prepare src buffer if needed for overwrite_from_image
      let srcData = null;
      if(op === 'overwrite_from_image' && state.srcLoaded){
        const sctx = state.srcOffctx;
        state.srcOffscreen.width = w;
        state.srcOffscreen.height = h;
        sctx.drawImage(state.srcImg, 0, 0, w, h);
        srcData = sctx.getImageData(0, 0, w, h).data;
      }

  // Minimal implementations for common ops in RGB/HSV/OKLAB.
      // For parity with backend: multiply/divide as raw factor; truncate step uses inverted normalized weight; contrast piecewise.

  function clamp01(x){ return Math.min(1, Math.max(0, x)); }
  function wrap01(x){ x = x % 1; return x < 0 ? x + 1 : x; }
      function rgb2hsv(r,g,b){
        const max = Math.max(r,g,b), min = Math.min(r,g,b);
        const v = max;
        const d = max - min;
        const s = max ? d / max : 0;
        let h = 0;
        if(d !== 0){
          if(max === r){ h = ((g - b) / d) % 6; }
          else if(max === g){ h = (b - r) / d + 2; }
          else { h = (r - g) / d + 4; }
          h /= 6; if(h < 0) h += 1;
        }
        return [h,s,v];
      }
      function hsv2rgb(h,s,v){
        const i = Math.floor(h*6);
        const f = h*6 - i;
        const p = v*(1-s);
        const q = v*(1-f*s);
        const t = v*(1-(1-f)*s);
        switch(i % 6){
          case 0: return [v,t,p];
          case 1: return [q,v,p];
          case 2: return [p,v,t];
          case 3: return [p,q,v];
          case 4: return [t,p,v];
          case 5: return [v,p,q];
        }
      }

      // --- Oklab conversions ---
      function srgbToLinear(c){
        return c <= 0.04045 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4);
      }
      function linearToSrgb(c){
        return c <= 0.0031308 ? 12.92*c : 1.055*Math.pow(Math.max(0,c), 1/2.4) - 0.055;
      }
      function rgb2oklab(r,g,b){
        // sRGB->linear
        const rl=srgbToLinear(r), gl=srgbToLinear(g), bl=srgbToLinear(b);
        const l = 0.4122214708*rl + 0.5363325363*gl + 0.0514459929*bl;
        const m = 0.2119034982*rl + 0.6806995451*gl + 0.1073969566*bl;
        const s = 0.0883024619*rl + 0.2817188376*gl + 0.6299787005*bl;
        const l3 = Math.cbrt(l), m3 = Math.cbrt(m), s3 = Math.cbrt(s);
        const L = 0.2104542553*l3 + 0.7936177850*m3 - 0.0040720468*s3;
        const A = 1.9779984951*l3 - 2.4285922050*m3 + 0.4505937099*s3;
        const B = 0.0259040371*l3 + 0.7827717662*m3 - 0.8086757660*s3;
        return [L,A,B];
      }
      function oklab2rgb(L,A,B){
        const l3 = L + 0.3963377774*A + 0.2158037573*B;
        const m3 = L - 0.1055613458*A - 0.0638541728*B;
        const s3 = L - 0.0894841775*A - 1.2914855480*B;
        const l = l3*l3*l3, m = m3*m3*m3, s = s3*s3*s3;
        const rl = 4.0767416621*l - 3.3077115913*m + 0.2309699292*s;
        const gl = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s;
        const bl = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s;
        const r = linearToSrgb(rl), g = linearToSrgb(gl), b = linearToSrgb(bl);
        return [clamp01(r), clamp01(g), clamp01(b)];
      }

      const amtNorm = amount255 / 255.0;

      function applyPixel(i){
        const r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
        if(op === 'invert'){
          const S = src.toLowerCase();
          if(S === 'rgb'){
            return [1-r, 1-g, 1-b];
          }else if(S === 'red' || S==='green' || S==='blue'){
            return [S==='red'?1-r:r, S==='green'?1-g:g, S==='blue'?1-b:b];
          }else if(S === 'hue' || S==='saturation' || S==='value' || S==='hsv'){
            const hsv = rgb2hsv(r,g,b);
            if(S==='hsv'){
              hsv[0] = 1-hsv[0]; hsv[1] = 1-hsv[1]; hsv[2] = 1-hsv[2];
            }else{
              const idx = {hue:0,saturation:1,value:2}[S];
              hsv[idx] = 1-hsv[idx];
            }
            return hsv2rgb(hsv[0],hsv[1],hsv[2]);
          }else if(S === 'oklab'){
            const lab = rgb2oklab(r,g,b);
            const L = 1 - clamp01(lab[0]);
            const a_n = clamp01((lab[1] + 0.4) / 0.8);
            const b_n = clamp01((lab[2] + 0.4) / 0.8);
            const a_i = (1 - a_n) * 0.8 - 0.4;
            const b_i = (1 - b_n) * 0.8 - 0.4;
            return oklab2rgb(L, a_i, b_i);
          }
        }
        // Overwrite operation: copy source channel/value into destination channel (cross-space allowed)
        if(op === 'overwrite' || op === 'overwrite_from_image'){
          const S = src.toLowerCase();
          const D = dst.toLowerCase();
          let rr=r, gg=g, bb=b;
          // Compute scalar source value in [0,1]
          let sr = r, sg = g, sb = b;
          if(op === 'overwrite_from_image' && srcData){
            sr = srcData[i] / 255; sg = srcData[i+1] / 255; sb = srcData[i+2] / 255;
          }
          // Fast path: group-to-group overwrite should copy full vector (parity with backend)
          if((S==='rgb' && D==='rgb')){
            return [sr, sg, sb];
          } else if((S==='hsv' && D==='hsv')){
            const [h,s,v] = rgb2hsv(sr,sg,sb);
            const rgb = hsv2rgb(h,s,v);
            return [rgb[0], rgb[1], rgb[2]];
          } else if((S==='oklab' && D==='oklab')){
            const [L,A,B_] = rgb2oklab(sr,sg,sb);
            const rgb = oklab2rgb(L,A,B_);
            return [rgb[0], rgb[1], rgb[2]];
          }
          let sval = 0;
          if(S==='red' || S==='green' || S==='blue'){
            sval = (S==='red') ? sr : (S==='green' ? sg : sb);
          } else if(S==='hue' || S==='saturation' || S==='value' || S==='hsv'){
            const [h,s,v] = rgb2hsv(sr,sg,sb);
            sval = (S==='hsv') ? h : (S==='hue' ? h : (S==='saturation' ? s : v));
          } else if(S==='oklab'){
            const lab = rgb2oklab(sr,sg,sb);
            sval = clamp01(lab[0]); // use L channel
          } else {
            // Default fallback mirrors backend: use red channel when source not a single channel
            sval = sr;
          }
          sval = clamp01(sval);

          // Apply to destination
          if(D==='red' || D==='green' || D==='blue'){
            if(D==='red') rr = sval;
            else if(D==='green') gg = sval;
            else bb = sval;
            return [rr,gg,bb];
          } else if(D==='hue' || D==='saturation' || D==='value'){
            let [h,s,v] = rgb2hsv(rr,gg,bb);
            if(D==='hue') h = sval; // backend clamps rather than wrapping
            else if(D==='saturation') s = sval;
            else v = sval;
            const rgb = hsv2rgb(h,s,v);
            return [rgb[0], rgb[1], rgb[2]];
          } else if(D==='rgb'){
            // Grayscale overwrite by scalar
            return [sval, sval, sval];
          } else if(D==='hsv'){
            // Set all HSV components from scalar; hue wraps [0,1), s and v clamp
            let h = sval - Math.floor(sval); // wrap
            let s2 = Math.max(0, Math.min(1, sval));
            let v2 = Math.max(0, Math.min(1, sval));
            const rgb = hsv2rgb(h, s2, v2);
            return [rgb[0], rgb[1], rgb[2]];
          } else if(D==='oklab'){
            // Set Oklab L directly; a and b via normalization [-0.4,0.4]
            const L = Math.max(0, Math.min(1, sval));
            const an = Math.max(0, Math.min(1, sval));
            const bn = Math.max(0, Math.min(1, sval));
            const a2 = an * 0.8 - 0.4;
            const b2 = bn * 0.8 - 0.4;
            const rgb = oklab2rgb(L, a2, b2);
            return [rgb[0], rgb[1], rgb[2]];
          }
          // If destination unsupported, return unchanged
          return [rr,gg,bb];
        }
        // Scalar ops subset: set/add/subtract/multiply/divide/clamp_min/clamp_max/truncate/contrast applied to source channel(s)
        const S = src.toLowerCase();
        let rr=r, gg=g, bb=b;
        function applyScalarRaw(x){
          if(op==='set') return amtNorm;
          if(op==='add') return x + amtNorm;
          if(op==='subtract') return x - amtNorm;
          if(op==='multiply') return x * Math.max(0, amount255);
          if(op==='divide') return x / Math.max(1e-8, amount255);
          if(op==='clamp_min') return Math.max(amtNorm, x);
          if(op==='clamp_max') return Math.min(amtNorm, x);
          if(op==='truncate'){
            const step = 1 - amtNorm; // inverted
            if(step <= 0) return x;
            return Math.round(x / step) * step;
          }
          if(op==='contrast'){
            const a = 1 - amtNorm; // inverted
            const k = 1 / Math.max(1e-6, 1 - a);
            let y = x;
            if(x > 0.5) y = x * k;
            else if(x < 0.5) y = x * (2 - k);
            // Backend clamps inside _apply_op for contrast
            return clamp01(y);
          }
          return x;
        }
        if(S==='rgb'){
          rr = clamp01(applyScalarRaw(rr)); gg = clamp01(applyScalarRaw(gg)); bb = clamp01(applyScalarRaw(bb));
        } else if(S==='red' || S==='green' || S==='blue'){
          if(S==='red') rr = clamp01(applyScalarRaw(rr));
          if(S==='green') gg = clamp01(applyScalarRaw(gg));
          if(S==='blue') bb = clamp01(applyScalarRaw(bb));
        } else if(S==='hue' || S==='saturation' || S==='value' || S==='hsv'){
          let [h,s,v] = rgb2hsv(rr,gg,bb);
          if(S==='hsv'){
            h = wrap01(applyScalarRaw(h)); s = clamp01(applyScalarRaw(s)); v = clamp01(applyScalarRaw(v));
          } else if(S==='hue'){
            h = wrap01(applyScalarRaw(h));
          } else if(S==='saturation'){
            s = clamp01(applyScalarRaw(s));
          } else if(S==='value'){
            v = clamp01(applyScalarRaw(v));
          }
          const rgb = hsv2rgb(h,s,v); rr=rgb[0]; gg=rgb[1]; bb=rgb[2];
        } else if(S==='oklab'){
          // Apply scalar op to all Oklab channels with a/b normalization for non-linear ops
          let [L,a,b_] = rgb2oklab(rr,gg,bb);
          function applyABNonLinear(val){
            const vn = clamp01((val + 0.4) / 0.8);
            const vn2 = clamp01(applyScalarRaw(vn));
            return vn2 * 0.8 - 0.4;
          }
          if(op==='set' || op==='contrast' || op==='truncate'){
            L = clamp01(applyScalarRaw(L));
            a = applyABNonLinear(a);
            b_ = applyABNonLinear(b_);
          } else {
            L = clamp01(applyScalarRaw(L));
            a = applyScalarRaw(a);
            b_ = applyScalarRaw(b_);
          }
          const rgb2 = oklab2rgb(L,a,b_);
          rr=rgb2[0]; gg=rgb2[1]; bb=rgb2[2];
        }
        return [rr,gg,bb];
      }

      for(let i=0;i<data.length;i+=4){
        const out = applyPixel(i);
        data[i] = Math.round(clamp01(out[0]) * 255);
        data[i+1] = Math.round(clamp01(out[1]) * 255);
        data[i+2] = Math.round(clamp01(out[2]) * 255);
        // alpha unchanged
      }
      ctx.putImageData(imgData, 0, 0);
      app.graph.setDirtyCanvas(true,true);
    }

    // Wire up events
    function bindChanges(){
      if(!node.widgets) return;
      node.widgets.forEach(w => {
        const orig = w.callback || w.onChange;
        const cb = function(){
          if(orig) try{ orig.apply(this, arguments); }catch(e){}
          renderEffect();
          app.graph.setDirtyCanvas(true,true);
        };
        w.callback = cb;
        w.onChange = cb;
      });
    }

    // Ensure a hidden 'preview_id' widget exists and is synced to node.id
    try {
      let w = getWidget('preview_id');
      if(!w){
        w = node.addWidget && node.addWidget("string", "preview_id", String(node.id ?? '0'), null, {serialize:true});
      }
      if(w){
        // hide widget completely
        w.computeSize = () => [0,0];
        w.draw = () => {};
        const setVal = (val) => {
          try{ w.value = String(val ?? ''); }catch(_){}
          if(Array.isArray(node.widgets_values)){
            const idx = node.widgets ? node.widgets.indexOf(w) : -1;
            if(idx >= 0) node.widgets_values[idx] = w.value;
          }
        };
        // Set immediately and re-sync shortly after creation to catch late id assignment
        setVal(node.id);
        setTimeout(() => setVal(node.id), 0);
      }
    } catch(e) { console.warn(`[${EXT_NAME}] preview_id widget setup failed`, e); }

  // Expose load functions so prototype wrappers can trigger loads
  node._channelOpsLoadBase = loadBase;
  node._channelOpsLoadBaseWithRetry = loadBaseWithRetry;
  node._channelOpsLoadBaseReset = resetPreview;
  node._channelOpsLoadSrc = loadSrc;
  node._channelOpsLoadSrcWithRetry = loadSrcWithRetry;

  bindChanges();
  loadBase();
  // Opportunistically try to load a secondary source preview if present
  loadSrc();
  }

  // removed: duplicate global loadSrc/loadSrcWithRetry (they now live inside ensureBehavior where 'state' exists)

app.registerExtension({
  name: EXT_NAME,
  async beforeQueuePrompt(data){
    // Ensure preview_id is correctly set to the node id in the prompt
    try {
      if(data && data.output){
        for(const [nid, entry] of Object.entries(data.output)){
          const ct = (entry && entry.class_type) ? String(entry.class_type) : '';
          if(ct.includes('ChannelOpsNode')){
            entry.inputs = entry.inputs || {};
            entry.inputs.preview_id = String(nid);
          }
        }
      }
    } catch(e) { console.warn(`[${EXT_NAME}] beforeQueuePrompt`, e); }
    return data;
  },
  async setup(){
    // Listen for backend push to refresh preview immediately after save
    try{
      api.addEventListener("channel_ops_preview", (ev) => {
        try{
          const detail = ev?.detail ?? ev;
          const pid = String(detail?.preview_id ?? "");
          const ts = (typeof detail?.ts === 'number') ? Math.floor(detail.ts * 1000) : Date.now();
          if(!pid) return;
          const nodes = (app?.graph?._nodes || []).filter(n => (n.comfyClass || n.type || '').toString().includes('ChannelOpsNode'));
          for(const n of nodes){
            try{
              // Ensure behavior attached
              if(!n._channelOpsPreviewAdded){ ensureBehavior(n); }
              // Match by preview_id widget value OR by node.id to avoid cross-node updates
              let w = null;
              if(n.widgets){ w = n.widgets.find(w => (w && (w.name === 'preview_id' || w.label === 'preview_id'))); }
              const widVal = w ? String(w.value ?? '') : '';
              const nodeIdOk = (typeof n.id === 'number' && n.id >= 0) ? String(n.id) : '';
              const matches = (widVal && widVal === pid) || (nodeIdOk && nodeIdOk === pid);
              if(matches){
                if(n._channelOpsLoadBaseReset) n._channelOpsLoadBaseReset();
                if(n._channelOpsLoadBaseWithRetry) n._channelOpsLoadBaseWithRetry(ts);
                else if(n._channelOpsLoadBase) n._channelOpsLoadBase(ts);
                // Also attempt to refresh potential secondary source preview
                if(n._channelOpsLoadSrcWithRetry) n._channelOpsLoadSrcWithRetry(ts);
                else if(n._channelOpsLoadSrc) n._channelOpsLoadSrc(ts);
              }
            }catch(e){}
          }
        }catch(e){ console.warn(`[${EXT_NAME}] event handler`, e); }
      });
    }catch(e){ console.warn(`[${EXT_NAME}] api event wiring`, e); }
    // Attach to any nodes already created (when loading a workflow)
    if(app?.graph?._nodes){
      for(const n of app.graph._nodes){
        if((n.comfyClass || n.type || '').toString().includes('ChannelOpsNode')){
          try{
            ensureBehavior(n);
            // Per-instance onExecuted wrapper to ensure retry even if prototype hijack misses
            const prev = n.onExecuted;
            n.onExecuted = function(){
              if(prev) try{ prev.apply(this, arguments); }catch(e){}
              try{
                if(this._channelOpsPreviewAdded){
                  if(this._channelOpsLoadBaseReset) this._channelOpsLoadBaseReset();
                  if(this._channelOpsLoadBaseWithRetry) this._channelOpsLoadBaseWithRetry();
                  else if(this._channelOpsLoadBase) this._channelOpsLoadBase();
                    // Also refresh potential secondary source preview
                    if(this._channelOpsLoadSrcWithRetry) this._channelOpsLoadSrcWithRetry();
                    else if(this._channelOpsLoadSrc) this._channelOpsLoadSrc();
                }
              }catch(e){}
            };
          }catch(e){ console.error(`[${EXT_NAME}] setup`, e); }
        }
      }
    }
  },
  async nodeCreated(node){
    if((node.comfyClass || node.type || '').toString().includes('ChannelOpsNode')){
      try{
        ensureBehavior(node);
        // Per-instance onExecuted wrapper
        const prev = node.onExecuted;
        node.onExecuted = function(){
          if(prev) try{ prev.apply(this, arguments); }catch(e){}
          try{
            if(this._channelOpsPreviewAdded){
              if(this._channelOpsLoadBaseReset) this._channelOpsLoadBaseReset();
              if(this._channelOpsLoadBaseWithRetry) this._channelOpsLoadBaseWithRetry();
              else if(this._channelOpsLoadBase) this._channelOpsLoadBase();
              // Also refresh potential secondary source preview
              if(this._channelOpsLoadSrcWithRetry) this._channelOpsLoadSrcWithRetry();
              else if(this._channelOpsLoadSrc) this._channelOpsLoadSrc();
            }
          }catch(e){}
        };
      }catch(e){ console.error(`[${EXT_NAME}] nodeCreated`, e); }
    }
  },
  async beforeRegisterNodeDef(nodeType, nodeData){
    // Be flexible about matching the node; some builds differ in name vs displayName
    const nm = (nodeData?.name || nodeType?.name || '').toString();
    if(nm.includes('ChannelOpsNode')){
      // Hijack lifecycle to guarantee our behavior on all instances
      const origCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function(){
        if(origCreated) try{ origCreated.apply(this, arguments); }catch(e){}
        try{ ensureBehavior(this); }catch(e){ console.error(`[${EXT_NAME}] onNodeCreated`, e); }
      };
      const origExec = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function(){
        if(origExec) try{ origExec.apply(this, arguments); }catch(e){}
        try{
          if(this._channelOpsPreviewAdded){
            if(this._channelOpsLoadBaseReset) this._channelOpsLoadBaseReset();
            if(this._channelOpsLoadBaseWithRetry){ this._channelOpsLoadBaseWithRetry(); }
            else if(this._channelOpsLoadBase){ this._channelOpsLoadBase(); }
            // Also refresh potential secondary source preview
            if(this._channelOpsLoadSrcWithRetry){ this._channelOpsLoadSrcWithRetry(); }
            else if(this._channelOpsLoadSrc){ this._channelOpsLoadSrc(); }
          }
        }catch(e){}
      };
      const origDraw = nodeType.prototype.onDrawForeground;
      nodeType.prototype.onDrawForeground = function(ctx){
        if(origDraw) try{ origDraw.apply(this, arguments); }catch(e){}
        try{ if(this._channelOpsPreviewAdded){ drawPreview(this, ctx); } }catch(e){}
      };
      const origResize = nodeType.prototype.onResize;
      nodeType.prototype.onResize = function(size){
        if(origResize) try{ origResize.apply(this, arguments); }catch(e){}
        try{ app.graph.setDirtyCanvas(true,true); }catch(e){}
      };
    }
  }
});
