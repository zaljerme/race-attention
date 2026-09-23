import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from race_layer import RACEAttention
from race_layer_vectorized import VectorizedRACEAttention


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


torch.manual_seed(42)

device = torch.device("cpu")

dim = 128
heads = 4
L = 8
P = 4
beta = 10


standard = StandardAttention(
    dim,
    heads
).to(device)

original_race = RACEAttention(
    dim=dim,
    num_heads=heads,
    L=L,
    P=P,
    beta=beta
).to(device)

vectorized_race = VectorizedRACEAttention(
    dim=dim,
    num_heads=heads,
    L=L,
    P=P,
    beta=beta
).to(device)


# Make the two RACE versions identical
vectorized_race.load_state_dict(
    original_race.state_dict()
)


standard.eval()
original_race.eval()
vectorized_race.eval()


sequence_lengths = [
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384
]


def benchmark(model, x, runs=5):

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            model(x)

    times = []

    for _ in range(runs):

        start = time.perf_counter()

        with torch.no_grad():
            model(x)

        end = time.perf_counter()

        times.append(end - start)

    return sum(times) / len(times)


standard_times = []
original_times = []
vectorized_times = []


for N in sequence_lengths:

    print()
    print("=" * 50)
    print(f"Sequence length: {N}")

    x = torch.randn(
        1,
        N,
        dim,
        device=device
    )

    try:
        standard_time = benchmark(
            standard,
            x
        )
    except RuntimeError:
        standard_time = None

    original_time = benchmark(
        original_race,
        x
    )

    vectorized_time = benchmark(
        vectorized_race,
        x
    )


    standard_times.append(standard_time)
    original_times.append(original_time)
    vectorized_times.append(vectorized_time)


    if standard_time is not None:

        print(
            f"Standard:        "
            f"{standard_time * 1000:.3f} ms"
        )

        print(
            f"Original RACE:   "
            f"{original_time * 1000:.3f} ms"
        )

        print(
            f"Vectorized RACE: "
            f"{vectorized_time * 1000:.3f} ms"
        )

        print(
            f"Vectorized vs Standard: "
            f"{standard_time / vectorized_time:.2f}x"
        )

    else:

        print("Standard: failed / out of memory")

        print(
            f"Original RACE:   "
            f"{original_time * 1000:.3f} ms"
        )

        print(
            f"Vectorized RACE: "
            f"{vectorized_time * 1000:.3f} ms"
        )


valid_lengths = []
valid_standard = []

for N, t in zip(
    sequence_lengths,
    standard_times
):
    if t is not None:
        valid_lengths.append(N)
        valid_standard.append(t)


plt.plot(
    valid_lengths,
    valid_standard,
    marker="o",
    label="Standard Attention"
)

plt.plot(
    sequence_lengths,
    original_times,
    marker="o",
    label="Original RACE"
)

plt.plot(
    sequence_lengths,
    vectorized_times,
    marker="o",
    label="Vectorized RACE"
)


plt.xlabel("Sequence Length")
plt.ylabel("Forward Pass Time (seconds)")
plt.title("Attention Scaling Comparison")

plt.xscale("log", base=2)
plt.yscale("log", base=10)

plt.legend()
plt.grid()

plt.savefig(
    "scaling_all.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()