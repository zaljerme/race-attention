import math
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from race_attention import race_attention


def exact_angular_attention(Q, K, V, gamma):
    q = F.normalize(Q, dim=-1)
    k = F.normalize(K, dim=-1)

    cosine = (q @ k.T).clamp(-1 + 1e-6, 1 - 1e-6)

    similarity = (
        1 - torch.acos(cosine) / math.pi
    ) ** gamma

    weights = similarity / (
        similarity.sum(dim=-1, keepdim=True) + 1e-8
    )

    return weights @ V


torch.manual_seed(42)

N = 256
d = 64
P = 4

Q = F.normalize(torch.randn(N, d), dim=-1)
K = F.normalize(torch.randn(N, d), dim=-1)
V = torch.randn(N, d)

exact = exact_angular_attention(Q, K, V, gamma=P)

betas = [2, 10, 40]
L_values = [8, 32, 128, 512, 2048]

for beta in betas:

    runtimes = []
    errors = []

    for L in L_values:

        torch.manual_seed(100)

        start = time.perf_counter()

        approx = race_attention(
            Q,
            K,
            V,
            L=L,
            P=P,
            beta=beta
        )

        runtime = time.perf_counter() - start

        error = (
            torch.norm(approx - exact)
            / torch.norm(exact)
        ).item()

        runtimes.append(runtime)
        errors.append(error)

        print(
            f"beta={beta:2d} | "
            f"L={L:4d} | "
            f"time={runtime:.4f}s | "
            f"error={error:.4f}"
        )

    plt.plot(
        runtimes,
        errors,
        marker="o",
        label=f"beta={beta}"
    )

    for x, y, L in zip(runtimes, errors, L_values):
        plt.annotate(
            f"L={L}",
            (x, y),
            fontsize=8
        )


plt.xlabel("Runtime (seconds)")
plt.ylabel("Relative Error")
plt.title("RACE Speed vs Accuracy Tradeoff")
plt.legend()
plt.grid()

plt.savefig("speed_accuracy_tradeoff.png")
plt.show()