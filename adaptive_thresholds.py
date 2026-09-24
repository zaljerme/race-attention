import csv
import math
import time
import torch
import torch.nn.functional as F

from race_layer_chunked import ChunkedRACEAttention

torch.set_num_threads(6)

DIM = 128
HEADS = 4

LENGTHS = [1024, 2048, 4096]
L_VALUES = [1, 2, 4, 8, 16, 32]
TARGETS = [0.90, 0.95, 0.97, 0.98]


def exact_attention(model, x):
    B, N, C = x.shape
    head_dim = C // HEADS

    qkv = model.qkv(x)
    qkv = qkv.reshape(B, N, 3, HEADS, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) / math.sqrt(head_dim)

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    out = (
        out.transpose(1, 2)
        .contiguous()
        .reshape(B, N, C)
    )

    return model.proj(out)


def benchmark(model, x, repeats=5):
    model.eval()

    with torch.no_grad():
        model(x)

        times = []

        for _ in range(repeats):
            start = time.perf_counter()
            out = model(x)
            end = time.perf_counter()

            times.append((end - start) * 1000)

    return out, sorted(times)[len(times) // 2]


all_results = []
selection_results = []

print()
print("=" * 75)
print("ADAPTIVE RACE QUALITY / LATENCY SWEEP")
print("=" * 75)

for n in LENGTHS:

    print()
    print(f"Sequence length: {n}")
    print("-" * 75)

    torch.manual_seed(42)
    x = torch.randn(1, n, DIM)

    measurements = {}

    for L in L_VALUES:

        torch.manual_seed(123)

        model = ChunkedRACEAttention(
            dim=DIM,
            num_heads=HEADS,
            L=L,
            P=4,
            beta=2.0,
            chunk_size=min(8, L)
        )

        with torch.no_grad():
            reference = exact_attention(model, x)

        out, ms = benchmark(model, x)

        cosine = F.cosine_similarity(
            out.flatten(),
            reference.flatten(),
            dim=0
        ).item()

        error = (
            torch.norm(out - reference)
            / torch.norm(reference)
        ).item()

        measurements[L] = {
            "time": ms,
            "cosine": cosine,
            "error": error
        }

        all_results.append([
            n,
            L,
            ms,
            cosine,
            error
        ])

        print(
            f"L={L:2d} | "
            f"{ms:7.2f} ms | "
            f"cosine={cosine:.4f} | "
            f"error={error:.4f}"
        )

    fixed_time = measurements[8]["time"]

    print()
    print("Adaptive selections:")
    print()

    for target in TARGETS:

        selected = None

        for L in L_VALUES:
            if measurements[L]["cosine"] >= target:
                selected = L
                break

        if selected is None:
            print(
                f"Target {target:.2f}: "
                f"No configuration met target"
            )
            continue

        data = measurements[selected]
        speedup = fixed_time / data["time"]

        print(
            f"Target {target:.2f} -> "
            f"L={selected:2d} | "
            f"cosine={data['cosine']:.4f} | "
            f"time={data['time']:.2f} ms | "
            f"vs L=8: {speedup:.2f}x"
        )

        selection_results.append([
            n,
            target,
            selected,
            data["cosine"],
            data["error"],
            data["time"],
            fixed_time,
            speedup
        ])


with open(
    "adaptive_threshold_measurements.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sequence_length",
        "L",
        "time_ms",
        "cosine",
        "relative_error"
    ])

    writer.writerows(all_results)


with open(
    "adaptive_threshold_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "sequence_length",
        "target_cosine",
        "selected_L",
        "actual_cosine",
        "relative_error",
        "adaptive_time_ms",
        "fixed_L8_time_ms",
        "speedup_vs_L8"
    ])

    writer.writerows(selection_results)


print()
print("=" * 75)
print("Saved:")
print("adaptive_threshold_measurements.csv")
print("adaptive_threshold_results.csv")
print("=" * 75)