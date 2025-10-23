// Layer Blending - In-node live preview (canvas)
// Displays the blended result of two images with selected blend mode and opacity.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const EXT_NAME = "Layer_Blending.Preview";

(function(){
  function drawPreview(node, ctx){
    const state = node._layerBlendState;
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
    if(node._layerBlendPreviewAdded) return;
    node._layerBlendPreviewAdded = true;

    const state = node._layerBlendState = {
      bgImg: new Image(),
      fgImg: new Image(),
      bgReady: false,
      fgReady: false,
      bgUrl: null,
      fgUrl: null,
      loadToken: 0,
      outCanvas: document.createElement('canvas'),
      outCtx: null,
    };
    state.outCtx = state.outCanvas.getContext('2d');

    function revokeUrl(kind){
      try{
        if(kind==='bg' && state.bgUrl){ URL.revokeObjectURL(state.bgUrl); state.bgUrl = null; }
        if(kind==='fg' && state.fgUrl){ URL.revokeObjectURL(state.fgUrl); state.fgUrl = null; }
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
      const base = [
        `/extensions/ComfyUI_Channel_Ops/layer_blend_bg_${pid}.png${q}`,
        `/extensions/Channel_Ops/layer_blend_bg_${pid}.png${q}`,
        `/extensions/ChannelOps/layer_blend_bg_${pid}.png${q}`,
      ];
      const top = [
        `/extensions/ComfyUI_Channel_Ops/layer_blend_fg_${pid}.png${q}`,
        `/extensions/Channel_Ops/layer_blend_fg_${pid}.png${q}`,
        `/extensions/ChannelOps/layer_blend_fg_${pid}.png${q}`,
      ];
      return {base, top};
    }

    async function fetchAsImage(urls, kind, token){
      try{
        for(const u of urls){
          const resp = await fetch(u, {cache:'no-store'});
          if(resp.ok){
            const blob = await resp.blob();
            const obj = URL.createObjectURL(blob);
            revokeUrl(kind);
            if(kind==='bg'){ state.bgUrl = obj; state.bgImg.src = obj; }
            else { state.fgUrl = obj; state.fgImg.src = obj; }
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

      state.bgReady = false; state.fgReady = false;
      state.bgImg.onload = ()=>{ if(token!==state.loadToken) return; state.bgReady = true; render(); };
      state.fgImg.onload = ()=>{ if(token!==state.loadToken) return; state.fgReady = true; render(); };
      state.bgImg.onerror = ()=>{ if(token!==state.loadToken) return; state.bgReady = false; render(); };
      state.fgImg.onerror = ()=>{ if(token!==state.loadToken) return; state.fgReady = false; render(); };

      // fire and forget
      fetchAsImage(paths.base, 'bg', token);
      fetchAsImage(paths.top, 'fg', token);
    }

    function clamp01(x){ return Math.min(1, Math.max(0, x)); }
    function blendPixel(a, b, mode){
      // a,b in [0,1]
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
        case 'add': return a + b;
        case 'subtract': return a - b;
        case 'normal':
        default: return b;
      }
    }

    function render(){
      const opacity = (parseFloat(getVal('opacity')) || 0) / 255.0;
  const mode = String(getVal('mode') || 'Normal').toLowerCase();
      const ready = state.bgReady && state.fgReady;
      state.outReady = false;
      if(!ready){ app.graph.setDirtyCanvas(true,true); return; }
      const w = Math.max(1, Math.min(state.bgImg.naturalWidth, state.fgImg.naturalWidth));
      const h = Math.max(1, Math.min(state.bgImg.naturalHeight, state.fgImg.naturalHeight));
      state.outCanvas.width = w; state.outCanvas.height = h;
      const ctx = state.outCtx;

      // Draw inputs resampled into temp offscreens
      const cA = document.createElement('canvas'); cA.width=w; cA.height=h; const xA = cA.getContext('2d'); xA.drawImage(state.bgImg, 0,0,w,h);
      const cB = document.createElement('canvas'); cB.width=w; cB.height=h; const xB = cB.getContext('2d'); xB.drawImage(state.fgImg, 0,0,w,h);
      const aData = xA.getImageData(0,0,w,h).data;
      const bData = xB.getImageData(0,0,w,h).data;

      const out = ctx.createImageData(w, h);
      const odata = out.data;
      for(let i=0;i<odata.length;i+=4){
        const ar=aData[i]/255, ag=aData[i+1]/255, ab=aData[i+2]/255;
        const br=bData[i]/255, bg=bData[i+1]/255, bb=bData[i+2]/255;
        let rr, gg, bb2;
        const m = mode;
        if(m === 'darker color'){
          const suma = ar+ag+ab, sumb = br+bg+bb;
          if(sumb < suma){ rr=br; gg=bg; bb2=bb; } else { rr=ar; gg=ag; bb2=ab; }
        } else if(m === 'lighter color'){
          const suma = ar+ag+ab, sumb = br+bg+bb;
          if(sumb > suma){ rr=br; gg=bg; bb2=bb; } else { rr=ar; gg=ag; bb2=ab; }
        } else {
          rr = blendPixel(ar, br, m); gg = blendPixel(ag, bg, m); bb2 = blendPixel(ab, bb, m);
        }
        // opacity composite over A
        rr = clamp01(ar*(1-opacity) + rr*opacity);
        gg = clamp01(ag*(1-opacity) + gg*opacity);
        bb2 = clamp01(ab*(1-opacity) + bb2*opacity);
        odata[i] = Math.round(rr*255);
        odata[i+1] = Math.round(gg*255);
        odata[i+2] = Math.round(bb2*255);
        odata[i+3] = 255;
      }
      ctx.putImageData(out, 0, 0);
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
      const wMode = getWidget('mode');
      const wOpac = getWidget('opacity');
      if(wMode){ wMode.tooltip = wMode.description = 'Blend modes: Normal, Multiply, Screen, Overlay, Soft Light, Add, Subtract'; }
      if(wOpac){ wOpac.tooltip = wOpac.description = 'Opacity: 0–255 (0 transparent, 255 fully applied)'; }
    }

    setPreviewId();
    bindChanges();
    applyTooltips();
    loadBoth();

    node._layerBlendLoad = loadBoth;

    const prevExec = node.onExecuted;
    node.onExecuted = function(){ if(prevExec) try{ prevExec.apply(this, arguments); }catch(_e){} try{ loadBoth(); }catch(_e){} };
  }

  app.registerExtension({
    name: EXT_NAME,
    async setup(){
      try{
        api.addEventListener('layer_blend_preview', (ev)=>{
          const detail = ev?.detail ?? ev;
          const pid = String(detail?.preview_id || '');
          const ts = (typeof detail?.ts === 'number') ? Math.floor(detail.ts*1000) : Date.now();
          if(!pid) return;
          const nodes = (app?.graph?._nodes || []).filter(n => (n.comfyClass || n.type || '').toString().includes('LayerBlendingNode'));
          for(const n of nodes){
            try{
              if(!n._layerBlendPreviewAdded) ensureBehavior(n);
              let w = n.widgets ? n.widgets.find(w => w && (w.name==='preview_id' || w.label==='preview_id')) : null;
              const widVal = w ? String(w.value || '') : '';
              const nodeIdOk = (typeof n.id === 'number' && n.id >= 0) ? String(n.id) : '';
              const matches = (widVal && widVal===pid) || (nodeIdOk && nodeIdOk===pid);
              if(matches && n._layerBlendLoad) n._layerBlendLoad(ts);
            }catch(_e){}
          }
        });
      }catch(_e){}
    },
    async beforeRegisterNodeDef(nodeType, nodeData){
      const nm = (nodeData?.name || nodeType?.name || '').toString();
      if(nm.includes('LayerBlendingNode')){
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(){ if(origCreated) try{ origCreated.apply(this, arguments); }catch(_e){} try{ ensureBehavior(this); }catch(_e){} };
        const origDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx){ if(origDraw) try{ origDraw.apply(this, arguments); }catch(_e){} try{ if(this._layerBlendPreviewAdded) drawPreview(this, ctx); }catch(_e){} };
      }
    },
  });
})();
