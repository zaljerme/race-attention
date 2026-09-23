import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from race_attention import race_attention


def exact_angular_attention(Q, K, V, gamma):
    q = F.normalize(Q, dim=-1)
    k = F.normalize(K, dim=-1)

    cosine = q @ k.T
    cosine = cosine.clamp(-1 + 1e-6, 1 - 1e-6)

    similarity = (
        1 - torch.acos(cosine) / math.pi
    ) ** gamma

    weights = similarity / (
        similarity.sum(dim=-1, keepdim=True) + 1e-8
    )

    return weights @ V


torch.manual_seed(42)

N = 512
d = 64

P = 4
L = 128

Q = F.normalize(torch.randn(N, d), dim=-1)
K = F.normalize(torch.randn(N, d), dim=-1)
V = torch.randn(N, d)
exact = exact_angular_attention(
    Q,
    K,
    V,
    gamma=P
)

betas = [0.5, 1, 2, 5, 10, 20, 40]

errors = []
similarities = []

for beta in betas:

    trial_errors = []
    trial_similarities = []

    for trial in range(5):

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

        similarity = F.cosine_similarity(
            approx.flatten(),
            exact.flatten(),
            dim=0
        ).item()

        trial_errors.append(error)
        trial_similarities.append(similarity)

    avg_error = sum(trial_errors) / len(trial_errors)
    avg_similarity = sum(trial_similarities) / len(trial_similarities)

    errors.append(avg_error)
    similarities.append(avg_similarity)

    print(
        f"beta={beta:5.1f} | "
        f"Error={avg_error:.4f} | "
        f"Similarity={avg_similarity:.4f}"
    )


plt.plot(betas, errors, marker="o")

plt.xlabel("Beta")
plt.ylabel("Relative Error")
plt.title("RACE Soft-LSH Bias vs Beta")
plt.grid()

plt.savefig("beta_benchmark.png")
plt.show()