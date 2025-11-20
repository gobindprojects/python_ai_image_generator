# Fully Fixed Text-to-Image Generator (HF InferenceClient)
# Supports: Txt2Img, Img2Img, Seed, Steps, Guidance, Negative Prompt

import os
from datetime import datetime
from huggingface_hub import InferenceClient
from PIL import Image
import io

# ----------------- Models -----------------
MODELS = {
    "FLUX.1-schnell": "black-forest-labs/FLUX.1-schnell", 
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1", 
    "Stable Diffusion v1.4": "CompVis/stable-diffusion-v1-4",
    "OpenJourney v4": "prompthero/openjourney-v4",
    "FLUX.1-dev": "black-forest-labs/FLUX.1-dev" 
}

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

# ----------------- Image Generator -----------------
class ImageGenerator:

    def __init__(self, api_token: str, output_dir: str = "./generated_images"):
        self.api_token = api_token
        self.output_dir = output_dir
        self.client = None
        os.makedirs(output_dir, exist_ok=True)

    def _get_client(self):
        self.client = InferenceClient(api_key=self.api_token)
        return self.client

    def generate_image(self,
                       prompt: str,
                       model_id: str,
                       negative_prompt: str = None,
                       num_inference_steps: int = 28,
                       guidance_scale: float = 7.5,
                       width: int = 512,
                       height: int = 512,
                       seed: int = None,
                       init_image=None
                       ) -> tuple[bool, str, Image.Image]:
        try:
            client = self._get_client()

            # ----------------- IMG2IMG -----------------
            if init_image is not None:
                init_img = Image.open(init_image).convert("RGB")
                response = client.image_to_image(
                    prompt=prompt,
                    image=init_img,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=0.7,
                    seed=seed
                )
                img = response

            # ----------------- TEXT2IMG -----------------
            else:
                img = client.text_to_image(
                    prompt=prompt,
                    model=model_id,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    seed=seed
                )

            # ----------------- Save Image -----------------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"img_{timestamp}_{clean_prompt.replace(' ', '_')}.png"
            filepath = os.path.join(self.output_dir, filename)

            img.save(filepath)

            return True, f"✅ Image generated successfully! Saved as {filename}", img

        except Exception as e:
            return False, f"❌ Error: {str(e)}", None

# ----------------- Helper Functions -----------------
def validate_api_token(token: str) -> bool:
    return token is not None and token.strip() != ""


def get_example_prompts() -> list[str]:
    return [
        "a futuristic samurai standing on a neon-lit rooftop",
        "a magical forest with floating orbs",
        "a cyberpunk robot face made of chrome",
        "a warrior goddess in golden armor",
        "a floating castle above the clouds",
        "a retro 80s synthwave car driving at night",
    ]


def get_model_display_name(model_id: str) -> str:
    for name, mid in MODELS.items():
        if mid == model_id:
            return name
    return model_id
