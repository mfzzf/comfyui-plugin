import io
import os
import json
import openai
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from .modelverse_api.client import ModelverseClient

# Cache for models list
_cached_models = None

def get_openai_models(api_key: str) -> List[str]:
    """Fetch available models from OpenAI API"""
    global _cached_models
    
    # Return cached models if available
    if _cached_models is not None:
        return _cached_models
    
    try:
        # Use OpenAI client to get models
        client = openai.OpenAI(api_key=api_key)
        models_response = client.models.list()
        
        # Filter for chat models and sort
        chat_models = [
            model.id for model in models_response.data 
            if 'gpt' in model.id.lower() or 'o1' in model.id.lower() or 'chatgpt' in model.id.lower()
        ]
        chat_models.sort()
        
        # Cache the result
        _cached_models = chat_models if chat_models else ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        return _cached_models
    except Exception as e:
        print(f"Failed to fetch models from OpenAI: {e}")
        # Return default models if API call fails
        default_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        _cached_models = default_models
        return default_models


class OpenAIChat:
    """OpenAI Chat node with support for text, images, and files"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get default models (will be updated when API key is provided)
        default_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (default_models, {"default": "gpt-4o-mini"}),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "What can you tell me about this?"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 128000
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "image_in": ("IMAGE", {}),
                "file_content": ("STRING", {
                    "multiline": True,
                    "tooltip": "Text content from a file to include in the conversation"
                }),
                "response_format": (["text", "json_object"], {"default": "text"}),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "chat"

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
        openai_client = openai.OpenAI(api_key=modelverse_client.api_key)
        
        # Update models list with actual available models
        available_models = get_openai_models(modelverse_client.api_key)
        
        # Check if the selected model is available
        if model not in available_models:
            print(f"Warning: Model '{model}' may not be available. Available models: {available_models}")
        
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


class OpenAICaptionImage:
    """Legacy node for image captioning - kept for compatibility"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "image_in": ("IMAGE", {}),
                "model": (["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], {"default": "gpt-4o-mini"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant."}),
                "caption_prompt": ("STRING", {"default": "What's in this image?"}),
                "max_tokens": ("INT", {"default": 300}),
                "temperature": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "caption"

    def caption(self, client, image_in, model, system_prompt, caption_prompt, max_tokens, temperature):
        # Create ModelverseClient and get API key
        api_key = client.get("api_key")
        if not api_key:
            raise ValueError("No API key found in the client")
            
        # Create ModelverseClient instance to get the actual API key
        modelverse_client = ModelverseClient(api_key)
        
        # Convert tensor to PIL Image
        pil_image = Image.fromarray(np.clip(255. * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Set up OpenAI client with ModelverseClient API key
        openai_client = openai.OpenAI(api_key=modelverse_client.api_key)

        # Make API call to OpenAI
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if response.choices[0].message.content is None:
            raise ValueError("No content in response")

        # Extract and return the caption
        caption = response.choices[0].message.content.strip()
        return (caption,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenAIChat": OpenAIChat,
    "OpenAICaptionImage": OpenAICaptionImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIChat": "OpenAI Chat",
    "OpenAICaptionImage": "OpenAI Caption Image",
}
