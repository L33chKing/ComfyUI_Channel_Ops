import os
from typing import Dict, Any
from .channel_ops import ChannelOpsNode
from .layer_blending import LayerBlendingNode

# Expose web extension directory so ComfyUI serves JS files from ./web
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "ChannelOpsNode": ChannelOpsNode,
    "LayerBlendingNode": LayerBlendingNode,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "ChannelOpsNode": "Channel Ops (live-preview)",
    "LayerBlendingNode": "Layer Blending (live-preview)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

# Cleanup: remove all preview images in our web folder at startup
def _cleanup_web_folder():
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        web_dir = os.path.join(this_dir, "web")
        if not os.path.isdir(web_dir):
            return
        for name in os.listdir(web_dir):
            # Only delete image files to avoid removing JS or other assets
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    os.remove(os.path.join(web_dir, name))
                except Exception:
                    pass
    except Exception:
        # Silent by design; preview cleanup is best-effort
        pass

_cleanup_web_folder()
