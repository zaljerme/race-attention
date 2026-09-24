import math
import time
import torch
import torch.nn.functional as F

from race_layer_chunked import ChunkedRACEAttention

torch.set_num_threads(6)

DIM = 128
HEADS = 4

L_VALUES = [1, 2, 4, 8, 16, 32]
TARGETS = [0.90, 0.95, 0.97, 0.98]

SEQ_LEN = 2048
CALIBRATION_SAMPLES = 5
TEST_SAMPLES = 10


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


def make_model(L):
    torch.manual_seed(123)

    return ChunkedRACEAttention(
        dim=DIM,
        num_heads=HEADS,
        L=L,
        P=4,
        beta=2.0,
        chunk_size=min(8, L)
    )


def measure_quality(L, x):
    model = make_model(L)
    model.eval()

    with torch.no_grad():
        ref = exact_attention(model, x)
        out = model(x)

    cosine = F.cosine_similarity(
        out.flatten(),
        ref.flatten(),
        dim=0
    ).item()

    return cosine


def measure_time(L, x, repeats=5):
    model = make_model(L)
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


print()
print("=" * 72)
print("OFFLINE-CALIBRATED ADAPTIVE RACE")
print("=" * 72)

# ============================================================
# CALIBRATION
# ============================================================

quality_by_L = {}

for L in L_VALUES:
    scores = []

    for i in range(CALIBRATION_SAMPLES):
        torch.manual_seed(1000 + i)

        x = torch.randn(
            1,
            SEQ_LEN,
            DIM
        )

        scores.append(
            measure_quality(L, x)
        )

    mean_cosine = sum(scores) / len(scores)
    quality_by_L[L] = mean_cosine

    print(
        f"L={L:2d} | "
        f"calibration cosine={mean_cosine:.4f}"
    )


# ============================================================
# SELECT L FOR EACH TARGET
# ============================================================

selected = {}

print()
print("Selected policies:")
print()

for target in TARGETS:

    selected_L = None

    for L in L_VALUES:
        if quality_by_L[L] >= target:
            selected_L = L
            break

    selected[target] = selected_L

    print(
        f"Target {target:.2f} -> "
        f"L={selected_L}"
    )


# ============================================================
# UNSEEN TEST DATA
# ============================================================

print()
print("=" * 72)
print("UNSEEN TEST RESULTS")
print("=" * 72)

for target in TARGETS:

    L = selected[target]

    scores = []

    for i in range(TEST_SAMPLES):
        torch.manual_seed(5000 + i)

        x = torch.randn(
            1,
            SEQ_LEN,
            DIM
        )

        scores.append(
            measure_quality(L, x)
        )

    mean_score = sum(scores) / len(scores)
    min_score = min(scores)

    torch.manual_seed(9999)

    x_time = torch.randn(
        1,
        SEQ_LEN,
        DIM
    )

    adaptive_time = measure_time(
        L,
        x_time
    )

    fixed_time = measure_time(
        8,
        x_time
    )

    speedup = fixed_time / adaptive_time

    print()
    print(
        f"Target {target:.2f}"
    )
    print(
        f"Selected L:      {L}"
    )
    print(
        f"Mean cosine:     {mean_score:.4f}"
    )
    print(
        f"Worst cosine:    {min_score:.4f}"
    )
    print(
        f"Runtime:         {adaptive_time:.2f} ms"
    )
    print(
        f"vs fixed L=8:    {speedup:.2f}x"
    )