"""MiniMax H3 Live Preview -- watch an H3 generation build up, mid-sampling, on the node."""

import logging

from comfy_api.latest import ComfyExtension, io

from .preview import MiniMaxH3LivePreview

logger = logging.getLogger(__name__)

WEB_DIRECTORY = "./web"


class MiniMaxH3PreviewExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MiniMaxH3LivePreview]


async def comfy_entrypoint() -> MiniMaxH3PreviewExtension:
    return MiniMaxH3PreviewExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
