import io
import os
import json
import openai
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from .modelverse_api.client import ModelverseClient
from comfy.comfy_types.node_typing import IO

# Hardcoded models list
DEFAULT_MODELS = [
    'ByteDance/doubao-1.5-thinking-vision-pro',
    'ByteDance/doubao-seed-1.6',
    'ByteDance/doubao-seed-1.6-thinking',
    'Qwen/QwQ-32B',
    'Qwen/Qwen-Image',
    'Qwen/Qwen-Image-Edit',
    'Qwen/Qwen3-235B-A22B',
    'Qwen/Qwen3-30B-A3B',
    'Qwen/Qwen3-32B',
    'Qwen/Qwen3-Coder',
    'Wan-AI/Wan2.2-I2V',
    'Wan-AI/Wan2.2-T2V',
    'baidu/ernie-4.5-turbo-128k',
    'baidu/ernie-4.5-turbo-vl-32k',
    'baidu/ernie-x1-turbo-32k',
    'black-forest-labs/flux-kontext-max',
    'black-forest-labs/flux-kontext-max/multi',
    'black-forest-labs/flux-kontext-max/text-to-image',
    'black-forest-labs/flux-kontext-pro',
    'black-forest-labs/flux-kontext-pro/multi',
    'black-forest-labs/flux-kontext-pro/text-to-image',
    'black-forest-labs/flux.1-dev',
    'claude-4-opus',
    'claude-4-sonnet',
    'deepseek-ai/DeepSeek-Prover-V2-671B',
    'deepseek-ai/DeepSeek-R1',
    'deepseek-ai/DeepSeek-R1-0528',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'deepseek-ai/DeepSeek-V3-0324',
    'deepseek-ai/DeepSeek-V3.1',
    'gemini-2.5-flash',
    'gemini-2.5-flash-image',
    'gemini-2.5-pro',
    'gpt-4.1-mini',
    'grok-4',
    'moonshotai/Kimi-K2-Instruct',
    'openai/gpt-4.1',
    'openai/gpt-5',
    'openai/gpt-5-mini',
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    'qwen/qwen2.5-vl-72b-instruct',
    'stepfun-ai/step1x-edit',
    'zai-org/glm-4.5',
    'zai-org/glm-4.5v'
]

# Default selected model
DEFAULT_MODEL = "gemini-2.5-flash"


class OpenAIChat:
    """OpenAI Chat node with support for text, images, and files"""
    
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (DEFAULT_MODELS, {"default": DEFAULT_MODEL}),
                "user_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "What can you tell me about this?"
                }),
                "temperature": (IO.FLOAT, {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "max_tokens": (IO.INT, {
                    "default": 1000,
                    "min": 1,
                    "max": 128000
                }),
                "top_p": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "system_prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "image_in": (IO.IMAGE, {}),
                "file_content": (IO.STRING, {
                    "multiline": True,
                    "tooltip": "Text content from a file to include in the conversation"
                }),
                "response_format": (["text", "json_object"], {"default": "text"}),
                "presence_penalty": (IO.FLOAT, {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "frequency_penalty": (IO.FLOAT, {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("response",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "chat"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Tell ComfyUI that this node's inputs may change"""
        return "static"

    def chat(self, 
             client: Dict[str, str],
             model: str,
             user_prompt: str,
             temperature: float,
             max_tokens: int,
             top_p: float,
             system_prompt: Optional[str] = "You are a helpful assistant.",
             image_in: Optional[Any] = None,
             file_content: Optional[str] = None,
             response_format: str = "text",
             presence_penalty: float = 0.0,
             frequency_penalty: float = 0.0) -> tuple:
        
        # Create ModelverseClient and get API key
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("No API key found in the client")
            
        # Create ModelverseClient instance to get the actual API key
        modelverse_client = ModelverseClient(api_key)
        
        # Initialize OpenAI client with the API key from ModelverseClient
        openai_client = openai.OpenAI(api_key=modelverse_client.api_key,base_url="https://api.modelverse.cn/v1")
        
        # Build messages
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message content
        user_content = []
        
        # Add main prompt
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Add file content if provided
        if file_content:
            file_text = f"\n\nFile content:\n{file_content}"
            user_content.append({
                "type": "text",
                "text": file_text
            })
        
        # Add image if provided
        if image_in is not None:
            # Convert tensor to PIL Image
            # Handle both single images and batches
            if len(image_in.shape) == 4:
                # Take first image from batch
                image_in = image_in[0]
            
            # Convert from tensor format (H, W, C) to numpy array
            image_array = image_in.cpu().numpy()
            
            # Ensure values are in 0-255 range
            image_array = np.clip(255. * image_array, 0, 255).astype(np.uint8)
            
            # Create PIL Image
            pil_image = Image.fromarray(image_array)
            
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Add image to content
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            })
        
        # Add user message with all content
        if len(user_content) == 1:
            # If only text, use simple format
            messages.append({
                "role": "user",
                "content": user_content[0]["text"]
            })
        else:
            # If multiple content types, use array format
            messages.append({
                "role": "user",
                "content": user_content
            })
        
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        # Add response format if JSON is requested
        if response_format == "json_object":
            api_params["response_format"] = {"type": "json_object"}
        
        try:
            # Make API call to OpenAI
            response = openai_client.chat.completions.create(**api_params)
            
            # Extract response content
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("No content in response")
                
                # Return the response as a tuple
                return (content.strip(),)
            else:
                raise ValueError("No choices in response")
                
        except Exception as e:
            error_msg = f"OpenAI API Error: {str(e)}"
            print(error_msg)
            return (error_msg,)



# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenAIChat": OpenAIChat,
    # "OpenAICaptionImage": OpenAICaptionImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIChat": "OpenAI Chat",
    # "OpenAICaptionImage": "OpenAI Caption Image",
}
