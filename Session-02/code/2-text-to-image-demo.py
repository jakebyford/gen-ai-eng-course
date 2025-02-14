# Install libraries
# pip install torch torchvision diffusers transformers accelerate matplotlib

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os


os.makedirs("text_to_image_output", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading Stable Diffusion Model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype = torch.float32 # USe float32 for CPU compatiblity
).to(device)

pipe.enable_attention_slicing()

prompts = [
    "An Avengers poster in futuristic style",
    "A realistic painting of Iron Man in space",
    "A mythical creature standing on a mountain", 
    "A serene landscape with Thor holding Mjolnir",
    "A comic book style image of Hulk smashing a wall"
]

# Generate and save images
for idx, prompt in enumerate(prompts):
    print(f"Generating image for prompt: {prompt}")
    try:
        image = pipe(prompt, height=256, width=256).images[0]  # Reduced resolution
        image_path = f"text_to_image_output/image_{idx+1}.png"
        image.save(image_path)
        print(f"Image saved to {image_path}")

    except RuntimeError as e:
        print(f"Error generating image for prompt '{prompt}': {e}")




