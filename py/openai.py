import io
import os
import json
import openai
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from .modelverse_api.client import ModelverseClient

# Default models list
DEFAULT_MODELS = [
    "openai/gpt-5",
    "openai/gpt-5-mini", 
    "openai/gpt-4.1",
    "gpt-4.1-mini",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen3-32B",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "claude-4-opus",
    "claude-4-sonnet",
    "grok-4"
]

# Default selected model
DEFAULT_MODEL = "gemini-2.5-flash"

# Cache for models list
_cached_models = None

def debug_cache_state():
    """Debug function to print cache state"""
    global _cached_models
    print(f"[DEBUG] Cache state: _cached_models = {_cached_models}")
    if _cached_models is not None:
        print(f"[DEBUG] Cache type: {type(_cached_models)}, length: {len(_cached_models)}")
        print(f"[DEBUG] First 3 models: {_cached_models[:3] if len(_cached_models) > 0 else 'Empty'}")
    else:
        print(f"[DEBUG] Cache is None")

def get_openai_models(api_key: str) -> List[str]:
    """Fetch available models from OpenAI API"""
    global _cached_models
    
    print(f"[DEBUG] get_openai_models called with api_key: {api_key[:10]}...")
    debug_cache_state()
    
    # Return cached models if available
    if _cached_models is not None:
        print(f"[DEBUG] Returning cached models: {len(_cached_models)} models")
        return _cached_models
    
    try:
        print(f"[DEBUG] Making API call to fetch models...")
        # Use OpenAI client to get models
        client = openai.OpenAI(api_key=api_key,base_url="https://api.modelverse.cn/v1")
        models_response = client.models.list()
        
        print(f"[DEBUG] API response received, processing models...")
        # Get all available models and sort
        chat_models = [model.id for model in models_response.data]
        chat_models.sort()
        
        print(f"[DEBUG] Found {len(chat_models)} models from API")
        print(f"[DEBUG] First 5 models: {chat_models[:5]}")
        
        # Cache the result
        _cached_models = chat_models if chat_models else DEFAULT_MODELS
        print(f"[DEBUG] Cached {len(_cached_models)} models")
        debug_cache_state()
        return _cached_models
    except Exception as e:
        print(f"[ERROR] Failed to fetch models from OpenAI: {e}")
        # Return default models if API call fails
        _cached_models = DEFAULT_MODELS
        print(f"[DEBUG] Set cache to DEFAULT_MODELS due to error")
        debug_cache_state()
        return DEFAULT_MODELS


class OpenAIChat:
    """OpenAI Chat node with support for text, images, and files"""
    
    @staticmethod
    def refresh_models(api_key: str):
        """Refresh the cached models list with the given API key"""
        global _cached_models
        print(f"[DEBUG] refresh_models called with api_key: {api_key[:10]}...")
        print(f"[DEBUG] Before refresh:")
        debug_cache_state()
        
        try:
            print(f"[DEBUG] Clearing cache...")
            _cached_models = None  # Clear cache first
            debug_cache_state()
            
            print(f"[DEBUG] Calling get_openai_models...")
            _cached_models = get_openai_models(api_key)
            print(f"[DEBUG] After get_openai_models:")
            debug_cache_state()
            
            print(f"Successfully refreshed models list: {len(_cached_models)} models available")
            return _cached_models
        except Exception as e:
            print(f"[ERROR] Failed to refresh models: {e}")
            _cached_models = DEFAULT_MODELS
            print(f"[DEBUG] Set to DEFAULT_MODELS due to error:")
            debug_cache_state()
            return DEFAULT_MODELS
    
    @classmethod
    def INPUT_TYPES(cls):
        # Dynamic models list that updates based on cache
        # This is called every time the node is created/refreshed in the UI
        global _cached_models
        
        print(f"[DEBUG] INPUT_TYPES called for OpenAIChat")
        print(f"[DEBUG] Current cache state in INPUT_TYPES:")
        debug_cache_state()
        
        # Use cached models if available, otherwise use defaults
        if _cached_models is not None and _cached_models != DEFAULT_MODELS:
            available_models = _cached_models
            print(f"[INFO] Using cached models: {len(available_models)} models available")
            print(f"[DEBUG] Cached models: {available_models}")
        else:
            available_models = DEFAULT_MODELS
            print(f"[INFO] Using default models: {len(available_models)} models")
            print(f"[DEBUG] Default models: {available_models}")
        
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "model": (available_models, {"default": DEFAULT_MODEL}),
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
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Tell ComfyUI that this node's inputs may change"""
        global _cached_models
        # Return a unique value to force refresh when models change
        if _cached_models is not None:
            return str(len(_cached_models))
        return "default"

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
        
        # Refresh models list if not cached yet or if this is the first time with this API key
        global _cached_models
        print(f"[DEBUG] In chat method, checking cache state:")
        debug_cache_state()
        
        if _cached_models is None or _cached_models == DEFAULT_MODELS:
            print("[INFO] Cache is empty or default, refreshing models list from API...")
            self.refresh_models(modelverse_client.api_key)
            print(f"[DEBUG] After refresh in chat method:")
            debug_cache_state()
        else:
            print(f"[INFO] Using existing cache with {len(_cached_models)} models")
        
        # Initialize OpenAI client with the API key from ModelverseClient
        openai_client = openai.OpenAI(api_key=modelverse_client.api_key,base_url="https://api.modelverse.cn/v1")
        
        # Model validation is handled at the input level, no need to check here
        
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
                "model": (DEFAULT_MODELS, {"default": DEFAULT_MODEL}),
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
        openai_client = openai.OpenAI(api_key=modelverse_client.api_key,base_url="https://api.modelverse.cn/v1")

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


class ModelListRefresh:
    """Helper node to refresh the available models list"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("MODELVERSE_API_CLIENT",),
                "refresh": ("BOOLEAN", {"default": True, "tooltip": "Click to refresh models list"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    CATEGORY = "UCLOUD_MODELVERSE"
    FUNCTION = "refresh_models_list"
    
    def refresh_models_list(self, client: Dict[str, str], refresh: bool) -> tuple:
        """Refresh the models list and return status"""
        print(f"[DEBUG] ModelListRefresh.refresh_models_list called with refresh={refresh}")
        
        if not refresh:
            return ("Models refresh skipped",)
            
        api_key = client.get("api_key")
        if not api_key:
            return ("Error: No API key found in the client",)
        
        print(f"[DEBUG] About to refresh with API key: {api_key[:10]}...")
        print(f"[DEBUG] Cache state before refresh:")
        debug_cache_state()
        
        try:
            # Create ModelverseClient instance
            modelverse_client = ModelverseClient(api_key)
            
            # Refresh models using OpenAIChat's method
            models = OpenAIChat.refresh_models(modelverse_client.api_key)
            
            print(f"[DEBUG] Cache state after refresh:")
            debug_cache_state()
            
            status_msg = f"✅ Successfully refreshed! Found {len(models)} models.\n"
            status_msg += f"Models: {', '.join(models[:10])}{'...' if len(models) > 10 else ''}\n"
            status_msg += "\n⚠️ IMPORTANT: Please refresh the OpenAI Chat node in ComfyUI to see the updated models list!"
            
            return (status_msg,)
            
        except Exception as e:
            print(f"[ERROR] Exception in refresh_models_list: {e}")
            return (f"❌ Error refreshing models: {str(e)}",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenAIChat": OpenAIChat,
    "OpenAICaptionImage": OpenAICaptionImage,
    "ModelListRefresh": ModelListRefresh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIChat": "OpenAI Chat",
    "OpenAICaptionImage": "OpenAI Caption Image",
    "ModelListRefresh": "🔄 Refresh Models List",
}
