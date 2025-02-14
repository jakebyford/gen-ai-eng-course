# pip install openai matplotlib pillow requests

import matplotlib.pyplot as plt
from PIL import Image
import requests
import os
from openai import OpenAI

api_key = "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"

client = OpenAI(
    api_key=api_key
)

os.makedirs("dalle_output", exist_ok=True)

prompts = [
    "A futuristic Iron Man suit flying over New York City",
    "An Avengers team poster in a cyberpunk universe",
    "Thor summoning lightning in a stormy sky",
    "A serene view of Wakanda with advanced technology in the background",
    "Captain America holding his shield in a desert landscape during sunset"
]

# def generate_image(prompts):
#     response = client.images.generate(
#         model="dall-e-3",
#         prompt=prompts,
#         n=1,
#         quality="standard",
#         size="512x512"
#     )

#     image_url = response.data[0].url
#     print(image_url)
#     image_data = requests.get(image_url).content
#     image_path = f"dalle_output/image_{idx+1}.png"

#     with open(image_path, "wb") as f:
#         f.write(image_data)


def generate_image(prompts):

    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt: {prompt}")
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                quality="standard"
            )

            image_url = response.data[0].url

            image_data = requests.get(image_url).content
            image_path = f"dalle_output/image_{idx+1}.png"

            with open(image_path, "wb") as f:
                f.write(image_data)

            image = Image.open(image_path)
            plt.imshow(image)
            plt.axis("off")
            plt.title(prompt)
            plt.show()

        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")

generate_image(prompts)

print(f"Images saved to 'dalle_output' directory")

