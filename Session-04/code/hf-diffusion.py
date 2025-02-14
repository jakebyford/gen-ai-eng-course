# pip install torch torchvision transformers diffusers

from diffusers import StableDiffusionPipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
).to(device)


# pipe = pipe.to("cuda")

# prompt = "A photo of a cat in the forest"

# image = pipe(prompt).images[0]

# image.save("cat.jpg")

# print("Image saved as cat.jpg")

def diffuser_interface(input_text):
    image = pipe(input_text).images[0]
    return image

import gradio as gr

interface = gr.Interface(
    fn=diffuser_interface,
    inputs=["text", "image"],
    outputs=["text", "label"],
    title="Multi Modality Demo"
)
    
interface.launch()