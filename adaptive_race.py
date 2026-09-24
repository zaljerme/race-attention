import csv
import time
import torch

from race_layer_chunked import ChunkedRACEAttention

torch.set_num_threads(6)
torch.manual_seed(42)

DIM = 128
HEADS = 4
LENGTHS = [1024, 2048, 4096, 8192, 16384]


def choose_L(sequence_length):
    if sequence_length <= 1024:
        return 2
    elif sequence_length <= 2048:
        return 4
    elif sequence_length <= 4096:
        return 8
    elif sequence_length <= 8192:
        return 12
    else:
        return 16


def benchmark(model, x, repeats=5):
    model.eval()

    with torch.no_grad():
        model(x)

        times = []

        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()

            times.append((end - start) * 1000)

    return sorted(times)[len(times) // 2]


results = []

print()
print("=" * 65)
print("ADAPTIVE RACE BENCHMARK")
print("=" * 65)

for n in LENGTHS:

    x = torch.randn(1, n, DIM)

    adaptive_L = choose_L(n)

    fixed = ChunkedRACEAttention(
        dim=DIM,
        num_heads=HEADS,
        L=8,
        P=4,
        beta=2.0,
        chunk_size=8
    )

    adaptive = ChunkedRACEAttention(
        dim=DIM,
        num_heads=HEADS,
        L=adaptive_L,
        P=4,
        beta=2.0,
        chunk_size=min(8, adaptive_L)
    )

    fixed_time = benchmark(fixed, x)
    adaptive_time = benchmark(adaptive, x)

    speedup = fixed_time / adaptive_time

    print()
    print(f"Sequence length: {n}")
    print(f"Adaptive L:     {adaptive_L}")
    print(f"Fixed L=8:      {fixed_time:.2f} ms")
    print(f"Adaptive RACE:  {adaptive_time:.2f} ms")
    print(f"Speedup:        {speedup:.2f}x")

    results.append({
        "sequence_length": n,
        "adaptive_L": adaptive_L,
        "fixed_L8_ms": fixed_time,
        "adaptive_ms": adaptive_time,
        "speedup": speedup
    })


with open("adaptive_race_results.csv", "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sequence_length",
            "adaptive_L",
            "fixed_L8_ms",
            "adaptive_ms",
            "speedup"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("=" * 65)
print("Saved to adaptive_race_results.csv")
print("=" * 65)