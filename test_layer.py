import torch
from race_layer import RACEAttention


model = RACEAttention(
    dim=128,
    num_heads=4,
    L=32,
    P=4,
    beta=10
)

x = torch.randn(
    8,      # batch
    64,     # tokens
    128     # embedding dimension
)

y = model(x)

print("Input: ", x.shape)
print("Output:", y.shape)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)