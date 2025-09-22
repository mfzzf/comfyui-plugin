from typing import Optional
from pydantic import Field
from ..utils import BaseRequest


class GPTImage1(BaseRequest):
    """
    gpt-image-1 text-to-image via /v1/images/generations.
    """

    API_PATH = "/v1/images/generations"

    prompt: str = Field(..., description="Prompt text")
    num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images (n)")
    size: Optional[str] = Field(default="1024x1024", description="Output size e.g. 1024x1024")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    response_format: Optional[str] = Field(default="url", description='"url" or "b64_json"')

    def __init__(
        self,
        prompt: str,
        num_images: Optional[int] = 1,
        size: Optional[str] = "1024x1024",
        # guidance_scale removed per API behavior
        negative_prompt: Optional[str] = "",
        response_format: Optional[str] = "url",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt = prompt
        self.num_images = num_images
        self.size = size
        self.negative_prompt = negative_prompt
        self.response_format = response_format

    def build_payload(self) -> dict:
        payload = {
            "model": "gpt-image-1",
            "prompt": self.prompt,
            "n": self.num_images,
            "size": self.size,
            "negative_prompt": self.negative_prompt,
            "response_format": self.response_format,
        }
        return self._remove_empty_fields(payload)

    def field_required(self):
        return ["prompt"]

    def field_order(self):
        return [
            "model",
            "prompt",
            "n",
            "size",
            "negative_prompt",
            "response_format",
        ]
