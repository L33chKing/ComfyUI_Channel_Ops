// Color Matching - In-node live preview (canvas).
// Recolors `image` so its color statistics match `reference`, using one of:
//   LAB       - mean/std transfer in the perceptual Oklab space (default)
//   RGB       - mean/std transfer directly in RGB
//   Histogram - per-channel cumulative-distribution matching
// The preview recomputes in JS from the two saved input previews so switching
// method updates instantly without re-running the graph.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const EXT_NAME = "Color_Matching.Preview";

(function(){
  // ---------- Oklab plane conversions (match channel_ops.py) ----------
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

  // ---------- statistics helpers ----------
  function meanStd(arr){
    const n = arr.length;
    if(n === 0) return [0, 0];
    let sum = 0;
    for(let i=0; i<n; i++) sum += arr[i];
    const mean = sum / n;
    let v = 0;
    for(let i=0; i<n; i++){ const d = arr[i] - mean; v += d*d; }
    return [mean, Math.sqrt(v / n)];
  }
  function transferPlane(src, refMean, refStd, srcMean, srcStd, out){
    const scale = refStd / (srcStd + 1e-6);
    for(let i=0; i<src.length; i++){
      out[i] = (src[i] - srcMean) * scale + refMean;
    }
  }
  // Per-channel histogram (CDF) matching. Builds a 256-entry LUT so mapped
  // source levels follow the reference's cumulative distribution.
  function histMatchChannel(src, ref, out){
    const sH = new Float64Array(256), rH = new Float64Array(256);
    for(let i=0; i<src.length; i++){ let q=(src[i]*255)|0; if(q<0)q=0; else if(q>255)q=255; sH[q]++; }
    for(let i=0; i<ref.length; i++){ let q=(ref[i]*255)|0; if(q<0)q=0; else if(q>255)q=255; rH[q]++; }
    let sa=0, ra=0;
    const sC = new Float64Array(256), rC = new Float64Array(256);
    for(let i=0; i<256; i++){ sa+=sH[i]; sC[i]=sa; }
    for(let i=0; i<256; i++){ ra+=rH[i]; rC[i]=ra; }
    const sd = sa || 1, rd = ra || 1;
    for(let i=0; i<256; i++){ sC[i]/=sd; rC[i]/=rd; }
    const lut = new Float32Array(256);
    let j = 0;
    for(let i=0; i<256; i++){
      while(j < 255 && rC[j] < sC[i]) j++;
      lut[i] = j / 255;
    }
    for(let i=0; i<out.length; i++){ let q=(src[i]*255)|0; if(q<0)q=0; else if(q>255)q=255; out[i]=lut[q]; }
  }

  function drawPreview(node, ctx){
    const state = node._colorMatchState;
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
      const msg = "Run node once to seed preview";
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

  function ensureBehavior(node){
    if(node._colorMatchPreviewAdded) return;
    node._colorMatchPreviewAdded = true;

    const state = node._colorMatchState = {
      srcImg: new Image(),
      refImg: new Image(),
      srcReady: false,
      refReady: false,
      srcUrl: null,
      refUrl: null,
      loadToken: 0,
      outCanvas: document.createElement('canvas'),
      outCtx: null,
      outReady: false,
    };
    state.outCtx = state.outCanvas.getContext('2d');

    function revokeUrl(kind){
      try{
        if(kind==='src' && state.srcUrl){ URL.revokeObjectURL(state.srcUrl); state.srcUrl = null; }
        if(kind==='ref' && state.refUrl){ URL.revokeObjectURL(state.refUrl); state.refUrl = null; }
      }catch(_){ }
    }

    function getWidget(name){
      if(!node.widgets) return null;
      return node.widgets.find(w => (w && (w.name===name || w.label===name)));
    }
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
      const src = [
        `/extensions/ComfyUI_Channel_Ops/color_match_src_${pid}.png${q}`,
        `/extensions/Channel_Ops/color_match_src_${pid}.png${q}`,
        `/extensions/ChannelOps/color_match_src_${pid}.png${q}`,
      ];
      const ref = [
        `/extensions/ComfyUI_Channel_Ops/color_match_ref_${pid}.png${q}`,
        `/extensions/Channel_Ops/color_match_ref_${pid}.png${q}`,
        `/extensions/ChannelOps/color_match_ref_${pid}.png${q}`,
      ];
      return {src, ref};
    }

    async function fetchAsImage(urls, kind, token){
      try{
        for(const u of urls){
          const resp = await fetch(u, {cache:'no-store'});
          if(resp.ok){
            const blob = await resp.blob();
            const obj = URL.createObjectURL(blob);
            revokeUrl(kind);
            if(kind==='src'){ state.srcUrl = obj; state.srcImg.src = obj; }
            else { state.refUrl = obj; state.refImg.src = obj; }
            return true;
          }
        }
      }catch(_){ }
      return false;
    }

    function loadBoth(tsOverride){
      const wid = getWidget('preview_id');
      const pid = String((wid && wid.value) || node.id || 'A').replace(/[^a-zA-Z0-9_-]/g, "_");
      const ts = (typeof tsOverride==='number' && isFinite(tsOverride)) ? tsOverride : Date.now();
      const paths = buildPaths(pid, ts);
      const token = ++state.loadToken;

      state.srcReady = false; state.refReady = false;
      state.srcImg.onload = ()=>{ if(token!==state.loadToken) return; state.srcReady = true; render(); };
      state.refImg.onload = ()=>{ if(token!==state.loadToken) return; state.refReady = true; render(); };
      state.srcImg.onerror = ()=>{ if(token!==state.loadToken) return; state.srcReady = false; render(); };
      state.refImg.onerror = ()=>{ if(token!==state.loadToken) return; state.refReady = false; render(); };

      fetchAsImage(paths.src, 'src', token);
      fetchAsImage(paths.ref, 'ref', token);
    }

    // Decode an <img> into R/G/B Float32 planes at its native size.
    function decodePlanes(img){
      const w = img.naturalWidth, h = img.naturalHeight;
      if(w < 1 || h < 1) return null;
      const c = document.createElement('canvas'); c.width = w; c.height = h;
      const x = c.getContext('2d'); x.drawImage(img, 0, 0, w, h);
      const data = x.getImageData(0,0,w,h).data;
      const n = w * h;
      const R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n);
      for(let i=0, p=0; p<n; p++, i+=4){
        R[p] = data[i]/255; G[p] = data[i+1]/255; B[p] = data[i+2]/255;
      }
      return { w, h, R, G, B };
    }

    function render(){
      const method = String(getVal('method') || 'LAB').toLowerCase();
      state.outReady = false;
      if(!(state.srcReady && state.refReady)){ app.graph.setDirtyCanvas(true,true); return; }
      const src = decodePlanes(state.srcImg);
      const ref = decodePlanes(state.refImg);
      if(!src || !ref){ app.graph.setDirtyCanvas(true,true); return; }

      const w = src.w, h = src.h, n = w * h;
      const oR = new Float32Array(n), oG = new Float32Array(n), oB = new Float32Array(n);

      if(method === 'histogram'){
        histMatchChannel(src.R, ref.R, oR);
        histMatchChannel(src.G, ref.G, oG);
        histMatchChannel(src.B, ref.B, oB);
      } else if(method === 'rgb'){
        const [srm, srs] = meanStd(src.R), [sgm, sgs] = meanStd(src.G), [sbm, sbs] = meanStd(src.B);
        const [rrm, rrs] = meanStd(ref.R), [rgm, rgs] = meanStd(ref.G), [rbm, rbs] = meanStd(ref.B);
        transferPlane(src.R, rrm, rrs, srm, srs, oR);
        transferPlane(src.G, rgm, rgs, sgm, sgs, oG);
        transferPlane(src.B, rbm, rbs, sbm, sbs, oB);
        for(let i=0; i<n; i++){
          if(oR[i]<0)oR[i]=0; else if(oR[i]>1)oR[i]=1;
          if(oG[i]<0)oG[i]=0; else if(oG[i]>1)oG[i]=1;
          if(oB[i]<0)oB[i]=0; else if(oB[i]>1)oB[i]=1;
        }
      } else {
        // LAB (Oklab) mean/std transfer.
        const sL = new Float32Array(n), sA = new Float32Array(n), sBl = new Float32Array(n);
        const rL = new Float32Array(n), rA = new Float32Array(n), rBl = new Float32Array(n);
        rgbToOklabPlanes(src.R, src.G, src.B, sL, sA, sBl);
        rgbToOklabPlanes(ref.R, ref.G, ref.B, rL, rA, rBl);
        const [slm, sls] = meanStd(sL), [sam, sas] = meanStd(sA), [sblm, sbls] = meanStd(sBl);
        const [rlm, rls] = meanStd(rL), [ram, ras] = meanStd(rA), [rblm, rbls] = meanStd(rBl);
        const wL = new Float32Array(n), wA = new Float32Array(n), wBl = new Float32Array(n);
        transferPlane(sL,  rlm,  rls,  slm,  sls,  wL);
        transferPlane(sA,  ram,  ras,  sam,  sas,  wA);
        transferPlane(sBl, rblm, rbls, sblm, sbls, wBl);
        oklabToRgbPlanes(wL, wA, wBl, oR, oG, oB);
      }

      state.outCanvas.width = w; state.outCanvas.height = h;
      const out = state.outCtx.createImageData(w, h);
      const od = out.data;
      for(let p=0, i=0; p<n; p++, i+=4){
        od[i]   = (oR[p]*255)|0;
        od[i+1] = (oG[p]*255)|0;
        od[i+2] = (oB[p]*255)|0;
        od[i+3] = 255;
      }
      state.outCtx.putImageData(out, 0, 0);
      state.outReady = true;
      app.graph.setDirtyCanvas(true,true);
    }

    function bindChanges(){
      if(!node.widgets) return;
      node.widgets.forEach(w => {
        const orig = w.callback || w.onChange;
        const cb = function(){ if(orig) try{ orig.apply(this, arguments); }catch(_e){} render(); };
        w.callback = cb; w.onChange = cb;
      });
    }

    function applyTooltips(){
      const wMethod = getWidget('method');
      if(wMethod){ wMethod.tooltip = wMethod.description = [
        'Color-matching method:',
        'LAB — mean/std transfer in perceptual Oklab space (recommended).',
        'RGB — mean/std transfer directly in RGB.',
        'Histogram — per-channel cumulative-distribution matching.',
      ].join('\n'); }
    }

    setPreviewId();
    bindChanges();
    applyTooltips();
    loadBoth();

    node._colorMatchLoad = loadBoth;

    const prevExec = node.onExecuted;
    node.onExecuted = function(){ if(prevExec) try{ prevExec.apply(this, arguments); }catch(_e){} try{ loadBoth(); }catch(_e){} };
  }

  app.registerExtension({
    name: EXT_NAME,
    async setup(){
      try{
        api.addEventListener('color_match_preview', (ev)=>{
          const detail = ev?.detail ?? ev;
          const pid = String(detail?.preview_id || '');
          const ts = (typeof detail?.ts === 'number') ? Math.floor(detail.ts*1000) : Date.now();
          if(!pid) return;
          const nodes = (app?.graph?._nodes || []).filter(n => (n.comfyClass || n.type || '').toString().includes('ColorMatchingNode'));
          for(const n of nodes){
            try{
              if(!n._colorMatchPreviewAdded) ensureBehavior(n);
              let w = n.widgets ? n.widgets.find(w => w && (w.name==='preview_id' || w.label==='preview_id')) : null;
              const widVal = w ? String(w.value || '') : '';
              const nodeIdOk = (typeof n.id === 'number' && n.id >= 0) ? String(n.id) : '';
              const matches = (widVal && widVal===pid) || (nodeIdOk && nodeIdOk===pid);
              if(matches && n._colorMatchLoad) n._colorMatchLoad(ts);
            }catch(_e){}
          }
        });
      }catch(_e){}
    },
    async beforeRegisterNodeDef(nodeType, nodeData){
      const nm = (nodeData?.name || nodeType?.name || '').toString();
      if(nm.includes('ColorMatchingNode')){
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(){ if(origCreated) try{ origCreated.apply(this, arguments); }catch(_e){} try{ ensureBehavior(this); }catch(_e){} };
        const origDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx){ if(origDraw) try{ origDraw.apply(this, arguments); }catch(_e){} try{ if(this._colorMatchPreviewAdded) drawPreview(this, ctx); }catch(_e){} };
      }
    },
  });
})();
