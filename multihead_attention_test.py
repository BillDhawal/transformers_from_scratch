import torch
import torch.nn as nn
import math
from model import MultiHeadAttention

# Sample Parameters
d_model = 512   # Embedding dimension
seq_len = 6     # Sequence length
batch_size = 2  # Number of sequences in the batch
num_heads = 8   # Number of attention heads
dropout_rate = 0.1  # Dropout rate

# Initialize MultiHeadAttention module
multihead_attention = MultiHeadAttention(d_model=d_model, h=num_heads, dropout=dropout_rate)

# Generate Sample Input Data
q = torch.rand(batch_size, seq_len, d_model)  # Query matrix (Batch, Seq_Len, d_model)
k = torch.rand(batch_size, seq_len, d_model)  # Key matrix (Batch, Seq_Len, d_model)
v = torch.rand(batch_size, seq_len, d_model)  # Value matrix (Batch, Seq_Len, d_model)
mask = None  # Optional mask (Batch, 1, Seq_Len, Seq_Len)

# Forward Pass through the MultiHeadAttention Module
output = multihead_attention(q, k, v, mask)

# Print Shapes of Output and Attention Scores
print("Output Shape:", output.shape)  # Expected: (Batch, Seq_Len, d_model)
print("Attention Scores Shape:", multihead_attention.attention_scores.shape)  # Expected: (Batch, num_heads, Seq_Len, Seq_Len)