# Env Creation
# pip installation

# CPU : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# GPU : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Colab !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

from transformers import pipeline
import requests
import gc

# === Local Hugging Face Pipeline ===
def loacl_pipeline_demo():
    # Explicitly specify model and enable GPU ifg available
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 # USe GPU; set to -1 for CPU
    )
    text = "Since December, sentiment towards small-cap stocks has weakened and cyclicals have lagged, said Steven G. DeSanctis, U.S. mid-cap strategist at Jefferies."
    result = classifier(text)
    print("Local Pipeline Output:", result)

# === Hugging Face Inference API ===
def hf_inference_api_demo():
    # API URL
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer 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"}
    text = "The new iPhone is amazing with its advanced camera features."
    candidate_labels = ["Technology", "Politics", "Sports"]

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": candidate_labels
        }
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    print("Inference API Output:", response.json())


if __name__ == "__main__":
    # Local Pipeline Demo
    loacl_pipeline_demo()

    # Hugging Face Inference API Demo
    hf_inference_api_demo()

    # Garbage Collection
    gc.collect()