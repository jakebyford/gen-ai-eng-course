import gradio as gr
from transformers import pipeline
from PIL import Image
import numpy as np


text_gen = pipeline("text-generation", model="gpt2", device=-1) # Use -1 for CPU, 0 for GPU

image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=-1)

def multi_modal_interface(input_text, input_image):
    
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

        text_output = text_gen(input_text, max_length=30, truncation=True)[0]["generated_text"]

        raw_image_output = image_classifier(input_image)

        image_output = {item['label']: item['score'] for item in raw_image_output}

        return text_output, image_output
    

interface = gr.Interface(
    fn=multi_modal_interface,
    inputs=["text", "image"],
    outputs=["text", "label"],
    title="Multi Modality Demo"
)
    
interface.launch()


