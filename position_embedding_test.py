import torch
from model import PositionalEmbedding  # Assuming PositionalEmbedding is saved in model.py

# Parameters
d_model = 512  # Embedding dimension
seq_len = 10   # Sequence length
dropout = 0.1  # Dropout rate

# Input Tensor (Batch of size 1, sequence length of 10, embedding size of d_model)
input_tensor = torch.zeros((1, seq_len, d_model))

# Instantiate PositionalEmbedding
pos_embedding = PositionalEmbedding(d_model=d_model, seq_len=seq_len, dropout=dropout)

# Apply Positional Embedding
output_tensor = pos_embedding(input_tensor)

# Print results
print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
print("Sample Positional Embedding:\n", output_tensor[0, :, :5])  # First sequence, first 5 dimensions