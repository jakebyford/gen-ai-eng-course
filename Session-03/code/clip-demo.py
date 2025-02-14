# Setup Env
# python -m venv venv
# source venv/bin/activate
# Windows - source venv\Scripts\activate
# pip install torch==1.7.1 torchvision diffusers transformers clip-by-openai matplotlib

import torch
from torchvision.transformers import Compose, Resize, CenterCrop, ToTensor, Normalize
from diffusers import DDPMPipeline, DDPMScheduler
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import os

os.makedirs("clip_output", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print("Loaded CLIP Model and Processor")

diffusion = DDPMPipeline.from_pretained("google/ddpm-celebanhq-256", use_safetensors=True).to(device)
scheduler = DDPMScheduler.from_config("google/ddpm-celebanhq-256")

prompt = "An epic scene of Thor with a futuristic Asgard in the background"

print("Encoding the text prompt with CLIP")

text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
text_features = clip_model.get_text_features(**text_inputs)

def preprocess_image(image):
    preprocess = Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return preprocess(image).unsqueeze(0).to(device)

def generate_clip_guided_image(prompt, num_inference_steps=50):
    latents = torch.randn((1, 3, 256, 256), device=device)

    for step in range(num_inference_steps):
        latents = latents.detach().requires_grad_()

        with torch.no_grad():
            noise_prediction = diffusion.unet(latents, step).sample
            latents = scheduler.step(model_output=noise_prediction, sample=latents, timestep=step).prev_sample

        with torch.no_grad():
            image = latents.detach().cpu()
            image = ( image / 2 + 0.5).clamp(0, 1)
            image = ( image * 255).type(torch.uint8)
            image = image.permute(0, 2, 3, 1).numpy()
            image_pil = diffusion.numpy_to_pil(image[0])

        image_tensor = preprocess_image(image_pil)
        image_feautures = clip_model.get_image_features(image_tensor)
        similarity = torch.cosine_similarity(image_feautures, text_features).mean()

        similarity.backward(retain_graph=True)

        if latents.grad is not None:
            grad = latents.grad.detach()
            latents = latents.detach() - 0.1 * grad

        if step % 10 == 0:
            img_path = f"clip_output/step_{step}.png"
            image_pil.save(img_path)
            print(f"Step {step} saved to {img_path}")

    final_path = "clip_output/final_image.png"
    image_pil.save(final_path)
    print(f"Final Image saved to {final_path}")
    return image_pil


final_image = generate_clip_guided_image(prompt)

# Display the image
plt.imshow(final_image)
plt.axis("off")
plt.title(f"Prompt: {prompt}") 
plt.show()
