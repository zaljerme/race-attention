import torch

from race_layer import RACEAttention
from race_layer_chunked import ChunkedRACEAttention


torch.manual_seed(42)

dim = 128
heads = 4
L = 8
P = 4
beta = 10

reference = RACEAttention(
    dim=dim,
    num_heads=heads,
    L=L,
    P=P,
    beta=beta
)

reference.eval()

x = torch.randn(2, 1024, dim)


with torch.no_grad():
    reference_output = reference(x)


for chunk_size in [1, 2, 4, 8]:

    model = ChunkedRACEAttention(
        dim=dim,
        num_heads=heads,
        L=L,
        P=P,
        beta=beta,
        chunk_size=chunk_size
    )

    # Give it exactly the same weights and hashes
    model.load_state_dict(
        reference.state_dict()
    )

    model.eval()

    with torch.no_grad():
        output = model(x)

    max_diff = (
        output - reference_output
    ).abs().max().item()

    mean_diff = (
        output - reference_output
    ).abs().mean().item()

    print(
        f"Chunk {chunk_size} | "
        f"Max diff: {max_diff:.10f} | "
        f"Mean diff: {mean_diff:.10f}"
    )