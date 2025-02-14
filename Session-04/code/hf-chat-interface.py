# Create V env
# python -m venv venv
# source venv/bin/activate
# pip install transformers datasets gradio torch torchvision tensorflow tf-keras



from transformers import pipeline
import gradio as gr
from datasets import load_dataset

print(f"============= Text Generation with GPT ==============")

text_gen = pipeline("text-generation", model="gpt2")
generated_text = text_gen("Hugging face is", max_length=20, num_return_sequences=1)

print(f"Generated Text: {generated_text}")



import gradio as gr

def chatbot(input_text):
    generated_text = text_gen(input_text, max_length=20, num_return_sequences=1)
    return generated_text[0]['generated_text']


print(f"============= Text Generation with Gradio ==============")


gr.Interface(fn=chatbot, inputs="image", outputs="text", title="Hugging Face Chatbot").launch()
