import torch
from model import InputEmbeddings  # Assuming the class is in model.py

# Sample vocabulary and token mapping
vocab = {"I": 0, "love": 1, "Transformers": 2, ".": 3}
vocab_size = len(vocab)
d_model = 512  # Embedding dimensions

# Tokenized input (manually mapped)
tokenized_input = ["I", "love", "Transformers", "."]
input_indices = [vocab[token] for token in tokenized_input]

# Convert to tensor for embedding layer
input_tensor = torch.tensor([input_indices])  # Shape: (1, 4) for one sequence

# Define InputEmbeddings layer
embedding_layer = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)

# Retrieve embeddings
output_embeddings = embedding_layer(input_tensor)

# Print results
print("Tokenized Input:", tokenized_input)
print("Token Indices:", input_indices)
print("Embeddings:\n", output_embeddings)
print("Output Shape:", output_embeddings.shape)