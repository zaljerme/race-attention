import csv
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
CALIBRATION_SAMPLES = 10
TEST_SAMPLES = 20


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
        reference = exact_attention(model, x)
        output = model(x)

    cosine = F.cosine_similarity(
        output.flatten(),
        reference.flatten(),
        dim=0
    ).item()

    return cosine


def measure_time(L, x, repeats=7):
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
print("=" * 75)
print("CONSERVATIVE OFFLINE-CALIBRATED ADAPTIVE RACE")
print("=" * 75)

calibration = {}

# ============================================================
# CALIBRATION
# Use WORST calibration score instead of mean for selection
# ============================================================

for L in L_VALUES:

    scores = []

    for i in range(CALIBRATION_SAMPLES):

        torch.manual_seed(1000 + i)

        x = torch.randn(
            1,
            SEQ_LEN,
            DIM
        )

        score = measure_quality(L, x)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    worst_score = min(scores)

    calibration[L] = {
        "mean": mean_score,
        "worst": worst_score
    }

    print(
        f"L={L:2d} | "
        f"mean={mean_score:.4f} | "
        f"worst={worst_score:.4f}"
    )


# ============================================================
# CONSERVATIVE POLICY SELECTION
# ============================================================

selected = {}

print()
print("Selected policies using worst-case calibration:")
print()

for target in TARGETS:

    selected_L = None

    for L in L_VALUES:

        if calibration[L]["worst"] >= target:
            selected_L = L
            break

    selected[target] = selected_L

    if selected_L is None:
        print(
            f"Target {target:.2f} -> "
            f"No L met the target"
        )
    else:
        print(
            f"Target {target:.2f} -> "
            f"L={selected_L}"
        )


# ============================================================
# FIXED L=8 TIMING BASELINE
# ============================================================

torch.manual_seed(9999)

x_time = torch.randn(
    1,
    SEQ_LEN,
    DIM
)

fixed_L8_time = measure_time(
    8,
    x_time
)


# ============================================================
# UNSEEN TEST SET
# ============================================================

results = []

print()
print("=" * 75)
print("UNSEEN TEST RESULTS")
print("=" * 75)

for target in TARGETS:

    L = selected[target]

    if L is None:
        continue

    scores = []

    for i in range(TEST_SAMPLES):

        torch.manual_seed(5000 + i)

        x = torch.randn(
            1,
            SEQ_LEN,
            DIM
        )

        score = measure_quality(L, x)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    worst_score = min(scores)

    adaptive_time = measure_time(
        L,
        x_time
    )

    speedup = fixed_L8_time / adaptive_time

    passed_mean = mean_score >= target
    passed_worst = worst_score >= target

    print()
    print(f"Target {target:.2f}")
    print(f"Selected L:         {L}")
    print(f"Mean cosine:        {mean_score:.4f}")
    print(f"Worst cosine:       {worst_score:.4f}")
    print(f"Mean meets target:  {passed_mean}")
    print(f"Worst meets target: {passed_worst}")
    print(f"Runtime:            {adaptive_time:.2f} ms")
    print(f"Fixed L=8 runtime:  {fixed_L8_time:.2f} ms")
    print(f"Speedup vs L=8:     {speedup:.2f}x")

    results.append([
        target,
        L,
        calibration[L]["mean"],
        calibration[L]["worst"],
        mean_score,
        worst_score,
        adaptive_time,
        fixed_L8_time,
        speedup,
        passed_mean,
        passed_worst
    ])


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "adaptive_conservative_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "target_cosine",
        "selected_L",
        "calibration_mean",
        "calibration_worst",
        "test_mean",
        "test_worst",
        "adaptive_time_ms",
        "fixed_L8_time_ms",
        "speedup_vs_L8",
        "mean_pass",
        "worst_pass"
    ])

    writer.writerows(results)


print()
print("=" * 75)
print("Saved to adaptive_conservative_results.csv")
print("=" * 75)