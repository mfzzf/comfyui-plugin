import base64
import torch
from typing import List
from comfy.comfy_types.node_typing import IO

from .modelverse_api.utils import imageurl2tensor, decode_image, images2tensor
from .modelverse_api.client import ModelverseClient
from .modelverse_api.requests.gpt_image_1 import GPTImage1


class GPTImage1Node:
    """
    gpt-image-1 text-to-image via Modelverse /v1/images/generations API.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "prompt": (IO.STRING, {"multiline": True, "default": "a beautiful flower"}),
                "size": (IO.STRING, {"default": "1024x1024", "tooltip": "e.g. 512x512, 1024x1024"}),
                "num_requests": (IO.INT, {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"}),
                "num_images": (IO.INT, {"default": 1, "min": 1, "max": 4, "step": 1, "display": "number"}),
            },
            "optional": {},
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "execute"

    async def execute(
        self,
        client,
        prompt: str,
        size: str = "1024x1024",
        num_requests: int = 1,
        num_images: int = 1,
    ):

        if not prompt:
            raise ValueError("Prompt is required")

        mv_client = ModelverseClient(client["api_key"])

        tasks = [
            mv_client.async_send_request(
                GPTImage1(
                    prompt=prompt,
                    num_images=num_images,
                    size=size,
                )
            )
            for i in range(num_requests)
        ]

        results = await mv_client.run_tasks(tasks)

        output_images_list: List[torch.Tensor] = []
        for data_list in results:
            if not data_list:
                print("WARN:", "No output in current request. Skipping...")
                continue

            # Default to URL output format per API
            output_images = imageurl2tensor(data_list)

            output_images_list.append(output_images)

        if not output_images_list:
            return (torch.zeros((1, 3, 1, 1)),)

        return (torch.cat(output_images_list, dim=0),)


NODE_CLASS_MAPPINGS = {
    "Modelverse GPTImage1Node": GPTImage1Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Modelverse GPTImage1Node": "Modelverse GPT Image 1",
}
