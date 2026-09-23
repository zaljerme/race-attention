import time
import torch

from race_layer_chunked import ChunkedRACEAttention


torch.manual_seed(42)

dim = 128
heads = 4
L = 8
P = 4
beta = 10

sequence_lengths = [1024, 2048, 4096, 8192, 16384]
chunk_sizes = [1, 2, 4, 8]


def benchmark(model, x, runs=5):
    model.eval()

    with torch.no_grad():
        for _ in range(2):
            model(x)

    times = []

    for _ in range(runs):
        start = time.perf_counter()

        with torch.no_grad():
            model(x)

        times.append(time.perf_counter() - start)

    return sum(times) / len(times)


for N in sequence_lengths:

    print()
    print("=" * 50)
    print(f"Sequence length: {N}")

    x = torch.randn(1, N, dim)

    for chunk_size in chunk_sizes:

        model = ChunkedRACEAttention(
            dim=dim,
            num_heads=heads,
            L=L,
            P=P,
            beta=beta,
            chunk_size=chunk_size
        )

        runtime = benchmark(model, x)

        print(
            f"Chunk size {chunk_size}: "
            f"{runtime * 1000:.3f} ms"
        )