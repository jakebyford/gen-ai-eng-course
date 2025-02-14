#
# pip install gensim numpy matplotlib scikit-learn torchvision transformers sentence-transformers
#

# 1. Word2Vec
# 2. GloVe
# 3. BERT
# 4. CLIP

import numpy as np
from gensim.models import Word2Vec

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print(f" >> ------------ Word2Vec Demo -----------")

corpus = [
    "The quick brown fox jumps over the lazy dog".split(),
    "The dog barked at the mailman".split(),
    "The fox ran into the forest".split(),
    "The mailman delivered the package to the house".split()
]

word2vec_model = Word2Vec(
    sentences=corpus, # Input data source
    vector_size=50, # Dimensionality of the Embedding
    window=2, # Maximum distance between the current and predicted word within a sentence
    sg=1, #skip-gram model --> 1-Skip Gram Model, 0 - CBOW Model
    min_count = 1, # Minimum number of times a word must appear in order to be included in the vocabulary
    epochs = 10
)

word2vec_model.save("word2vec_model.model")

similar_words = word2vec_model.wv.most_similar("dog", topn=3)
print(f"Most Similar Words to 'fox': {similar_words}")

print(f" >> ---------- GloVe Demo ----------------")

# Download GloVe embeddings from https://nlp.stanford.edu/projects/glove
glove_path = "glove.6B.50d.txt"

glove_embeddings = {}

with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype="float32")
        glove_embeddings[word] = embedding

sample_sentence = "The quick brown fox"
vectorized_sentence = np.mean([glove_embeddings.get(word, np.zeros(50)) for word in sample_sentence.split()], axis=0)

print(f"Vectorized Sentence: {vectorized_sentence}")


# Demo 3
print(f" >> --------- BERT DEMO ----------------")

bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = ["The quick brown fox", "A fast dark animal", "A package was delivered"]
sentence_embeddings = bert_model.encode(sentences)
similarity = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])
similarity_1 = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[2]])
print(f"Similarity between the two sentences: {similarity}")
print(f"Similarity between the two sentences: {similarity_1}")

# Demo 4
# CLIP Demo
print(f" >> ----------- CLIP Demo -----------")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("fox.jpg")
captions = ["A photo of a dog", "A photo of a fox", "A photo of a cat", "A photo of a package"]
inputs = clip_processor(text=captions, images=image, return_tensors="pt", padding=True)
outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image # Image to Text similarity scores
probs = logits_per_image.softmax(dim=1)

print(f"CLIP matching probabilities: {probs}")
probs_list = probs[0].tolist()

for captions, probs in zip(captions, probs_list):
    print(f"Caption: {captions}, Probability: {probs}")
