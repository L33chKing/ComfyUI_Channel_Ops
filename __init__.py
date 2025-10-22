from typing import Dict, Any
from .channel_ops import ChannelOpsNode

# Expose web extension directory so ComfyUI serves JS files from ./web
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "ChannelOpsNode": ChannelOpsNode,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "ChannelOpsNode": "Channel Ops (live-preview)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
