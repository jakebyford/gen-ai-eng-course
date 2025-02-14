# pip install torch torchvision diffusers transformers accelerate pillow matplotlib

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os


os.makedirs("stability_sd_output", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Stability AI's Stable Diffusion Model...")

pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
).to(device)

pipeline.enable_attention_slicing()  # Optimize memory for large models

prompts = [
    "A futuristic cityscape with flying cars and neon lights",
#     "Thor holding his hammer in a futuristic battle scene",
#     "Iron Man flying over a cyberpunk cityscape at night",
#     "Captain America leading an army in a desert",
#     "Black Widow in a stealth mission with a futuristic suit"
]

def generate_images(prompts, guidance_scale=7.5, num_inference_steps=50):
    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt: {prompt}")
        try:
            # Generate image
            image = pipeline(prompt, guidance_scale=guidance_scale, num_inference_steps= num_inference_steps).images[0]

            # Save and display the image
            image_path = f"stability_sd_output/image_{idx+1}.png"
            image.save(image_path)
            print(f"Image saved: {image_path}")

            # Display the image
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Prompt: {prompt}") 
            plt.show()

        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")

generate_images(prompts)
