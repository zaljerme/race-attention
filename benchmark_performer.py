import gc
import csv
import time
import torch
import torch.nn as nn
from performer_pytorch import SelfAttention
from race_layer_chunked import ChunkedRACEAttention

torch.set_num_threads(6)
torch.manual_seed(42)

DIM = 128
HEADS = 4
LENGTHS = [1024, 2048, 4096, 8192, 16384]


class StandardAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            DIM, HEADS, batch_first=True
        )

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


def benchmark(model, x, repeats=3):
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

for n in LENGTHS:
    print(f"\nSequence length: {n}")

    x = torch.randn(1, n, DIM)

    models = {
        "Standard": StandardAttention(),
        "RACE": ChunkedRACEAttention(
            dim=DIM,
            num_heads=HEADS,
            L=8,
            P=4,
            beta=2.0,
            chunk_size=8,
        ),
        "Performer": SelfAttention(
            dim=DIM,
            heads=HEADS,
            causal=False,
        ),
    }

    for name, model in models.items():
        gc.collect()

        ms = benchmark(model, x)

        print(f"{name:10s}: {ms:.2f} ms")

        results.append({
            "sequence_length": n,
            "method": name,
            "time_ms": ms,
        })

        del model
        gc.collect()


with open("performer_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sequence_length", "method", "time_ms"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved to performer_benchmark.csv")