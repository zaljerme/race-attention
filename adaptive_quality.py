import math
import time
import torch
import torch.nn.functional as F

from race_layer_chunked import ChunkedRACEAttention

torch.set_num_threads(6)
torch.manual_seed(42)

DIM = 128
HEADS = 4
L_VALUES = [1, 2, 4, 8, 16, 32]
LENGTHS = [1024, 2048, 4096]

TARGET_COSINE = 0.95


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


def benchmark(model, x, repeats=3):
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


print()
print("=" * 70)
print("ERROR-BUDGETED ADAPTIVE RACE")
print("=" * 70)

for n in LENGTHS:

    print()
    print(f"Sequence length: {n}")
    print("-" * 70)

    torch.manual_seed(42)
    x = torch.randn(1, n, DIM)

    best = None

    for L in L_VALUES:

        # Reset seed so every L gets identical learned projections
        torch.manual_seed(123)

        model = ChunkedRACEAttention(
            dim=DIM,
            num_heads=HEADS,
            L=L,
            P=4,
            beta=2.0,
            chunk_size=min(8, L),
        )

        with torch.no_grad():
            reference = exact_attention(model, x)

        out, ms = benchmark(model, x)

        cosine = F.cosine_similarity(
            out.flatten(),
            reference.flatten(),
            dim=0
        ).item()

        relative_error = (
            torch.norm(out - reference)
            / torch.norm(reference)
        ).item()

        print(
            f"L={L:2d} | "
            f"time={ms:7.2f} ms | "
            f"cosine={cosine:.4f} | "
            f"rel_error={relative_error:.4f}"
        )

        if best is None and cosine >= TARGET_COSINE:
            best = (L, ms, cosine, relative_error)

    print()

    if best:
        print(
            f"Selected L={best[0]} | "
            f"{best[1]:.2f} ms | "
            f"cosine={best[2]:.4f} | "
            f"error={best[3]:.4f}"
        )
    else:
        print("No L reached target cosine 0.95.")