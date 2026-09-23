import math
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

L_values = [8, 32, 128, 512, 2048]
betas = [2, 10, 40]

for beta in betas:

    errors = []

    for L in L_values:

        trial_errors = []

        for trial in range(3):

            torch.manual_seed(100 + trial)

            approx = race_attention(
                Q,
                K,
                V,
                L=L,
                P=P,
                beta=beta
            )

            error = (
                torch.norm(approx - exact)
                / torch.norm(exact)
            ).item()

            trial_errors.append(error)

        avg_error = sum(trial_errors) / len(trial_errors)
        errors.append(avg_error)

        print(
            f"beta={beta:2d} | "
            f"L={L:4d} | "
            f"error={avg_error:.4f}"
        )

    plt.plot(
        L_values,
        errors,
        marker="o",
        label=f"beta={beta}"
    )


plt.xscale("log", base=2)

plt.xlabel("Number of Hash Tables (L)")
plt.ylabel("Relative Error")
plt.title("RACE Bias-Variance Tradeoff")
plt.legend()
plt.grid()

plt.savefig("bias_variance.png")
plt.show()