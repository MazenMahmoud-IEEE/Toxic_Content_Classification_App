# imagecaption.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model & processor once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path: str) -> str:
    """
    Generate an image caption using BLIP.
    Args:
        image_path (str): Path to the uploaded image.
    Returns:
        str: Generated caption.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
