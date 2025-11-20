
"""
Text-to-Image generation helper functions using Hugging Face InferenceClient
This module provides the core functionality for image generation
"""

import os
from datetime import datetime
from huggingface_hub import InferenceClient
from PIL import Image
import streamlit as st

MODELS = {
    "FLUX.1-schnell": "black-forest-labs/FLUX.1-schnell", 
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1", 
    "Stable Diffusion v1.4": "CompVis/stable-diffusion-v1-4",
    "OpenJourney v4": "prompthero/openjourney-v4",
    "FLUX.1-dev": "black-forest-labs/FLUX.1-dev" 
}

# Model information for display
MODEL_INFO = {
    "black-forest-labs/FLUX.1-schnell": {
        "license": "Apache 2.0",
        "status": "✅ Fully Free",
        "description": "Fast generation with high quality"
    },
    "runwayml/stable-diffusion-v1-5": {
        "license": "CreativeML Open RAIL-M",
        "status": "✅ Generally Free",
        "description": "Most tested and reliable model"
    },
    "stabilityai/stable-diffusion-2-1": {
        "license": "CreativeML Open RAIL-M", 
        "status": "✅ Generally Free",
        "description": "Enhanced version with better prompt understanding"
    },
    "CompVis/stable-diffusion-v1-4": {
        "license": "CreativeML Open RAIL-M",
        "status": "✅ Generally Free", 
        "description": "Original stable diffusion model"
    },
    "prompthero/openjourney-v4": {
        "license": "CreativeML Open RAIL-M",
        "status": "✅ Free for most uses",
        "description": "Great for artistic and stylized images"
    },
    "black-forest-labs/FLUX.1-dev": {
        "license": "Non-commercial",
        "status": "⚠️ Personal Use Only",
        "description": "Premium quality, slower generation"
    }
}

class ImageGenerator:
    """Text-to-image generator using Hugging Face models"""
    
    def __init__(self, api_token: str, output_dir: str = "./generated_images"):
        """
        Initialize the image generator
        
        Args:
            api_token (str): Hugging Face API token
            output_dir (str): Directory to save generated images
        """
        self.api_token = api_token
        self.output_dir = output_dir
        self.client = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_client(self):
        """Get or create HuggingFace client"""
        if self.client is None:
            self.client = InferenceClient(api_key=self.api_token)
        return self.client
    
    def generate_image(self, prompt: str, model_id: str) -> tuple[bool, str, Image.Image]:
        """
        Generate image from text prompt
        
        Args:
            prompt (str): Text description for image generation
            model_id (str): Model ID to use for generation
            
        Returns:
            tuple: (success: bool, message: str, image: PIL.Image or None)
        """
        try:
            client = self._get_client()
            
            # Generate image
            image = client.text_to_image(prompt, model=model_id)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"img_{timestamp}_{clean_prompt.replace(' ', '_')}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            image.save(filepath)
            
            success_message = f"✅ Image generated successfully!\n📁 Saved as: {filename}\n📏 Size: {image.size}"
            return True, success_message, image
            
        except Exception as e:
            error_message = f"❌ Generation failed: {str(e)}"
            return False, error_message, None
    
    def get_model_info(self, model_id: str) -> dict:
        """Get information about a specific model"""
        return MODEL_INFO.get(model_id, {
            "license": "Unknown",
            "status": "❓ Check model page",
            "description": "No information available"
        })

def validate_api_token(api_token: str) -> bool:
    """
    Validate if the API token is set and not empty
    
    Args:
        api_token (str): API token to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return api_token is not None and api_token.strip() != ""

def get_example_prompts() -> list[str]:
    """Get a list of example prompts for inspiration"""
    return [
        "a red sports car on a mountain road at sunset",
        "a cute robot reading a book in a cozy library",
        "a magical forest with glowing mushrooms and fireflies",
        "a cyberpunk city street with neon lights at night",
        "a peaceful zen garden with koi pond and cherry blossoms",
        "a space astronaut floating among colorful nebulae",
        "a steampunk airship flying over Victorian London",
        "a dragon perched on a castle tower during a thunderstorm"
    ]

def get_model_display_name(model_id: str) -> str:
    """Get display name for a model ID"""
    for display_name, id_value in MODELS.items():
        if id_value == model_id:
            return display_name
    return model_id