import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from race_layer import RACEAttention


class StandardAttention(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x):
        out, _ = self.attn(
            x, x, x,
            need_weights=False
        )
        return out


device = torch.device("cpu")

dim = 128
heads = 4

standard = StandardAttention(dim, heads).to(device)

race = RACEAttention(
    dim=dim,
    num_heads=heads,
    L=8,
    P=4,
    beta=10
).to(device)


sequence_lengths = [
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096
]

standard_times = []
race_times = []


def benchmark(model, x, runs=5):

    model.eval()

    # warmup
    with torch.no_grad():
        model(x)

    times = []

    for _ in range(runs):

        start = time.perf_counter()

        with torch.no_grad():
            model(x)

        end = time.perf_counter()

        times.append(end - start)

    return sum(times) / len(times)


for N in sequence_lengths:

    print(f"\nSequence length: {N}")

    x = torch.randn(
        1,
        N,
        dim,
        device=device
    )

    standard_time = benchmark(
        standard,
        x
    )

    race_time = benchmark(
        race,
        x
    )

    standard_times.append(standard_time)
    race_times.append(race_time)

    speedup = standard_time / race_time

    print(
        f"Standard: {standard_time:.6f}s"
    )

    print(
        f"RACE:     {race_time:.6f}s"
    )

    print(
        f"Speedup:  {speedup:.2f}x"
    )


plt.plot(
    sequence_lengths,
    standard_times,
    marker="o",
    label="Standard Attention"
)

plt.plot(
    sequence_lengths,
    race_times,
    marker="o",
    label="RACE Attention"
)

plt.xlabel("Sequence Length")
plt.ylabel("Forward Pass Time (seconds)")
plt.title("Transformer Attention Scaling")

plt.xscale("log", base=2)

plt.legend()
plt.grid()

plt.savefig(
    "transformer_scaling.png"
)

plt.show()