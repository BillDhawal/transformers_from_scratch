import torch
from torch.nn import Module
from model import LayerNormalization, FeedForward

# Define sample input
batch_size = 3
seq_len = 5
d_model = 8
x = torch.randn(batch_size, seq_len, d_model)  # Random input tensor

# Layer Normalization Test
layer_norm = LayerNormalization(eps=1e-6)
normalized_output = layer_norm(x)
print("Layer Normalization Output:")
print(normalized_output)

# Feed Forward Test
d_ff = 16  # Hidden dimension for Feed Forward
dropout_rate = 0.1
feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout_rate)
ff_output = feed_forward(x)
print("\nFeed Forward Output:")
print(ff_output)