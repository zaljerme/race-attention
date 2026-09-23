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
beta = 5.0

Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)

exact = exact_angular_attention(Q, K, V, gamma=P)

table_counts = [1, 2, 4, 8, 16, 32]

errors = []
similarities = []

for L in table_counts:

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

        relative_error = (
            torch.norm(approx - exact)
            / torch.norm(exact)
        ).item()

        cosine_similarity = F.cosine_similarity(
            approx.flatten(),
            exact.flatten(),
            dim=0
        ).item()

        trial_errors.append(relative_error)
        trial_similarities.append(cosine_similarity)

    error = sum(trial_errors) / len(trial_errors)
    similarity = sum(trial_similarities) / len(trial_similarities)

    errors.append(error)
    similarities.append(similarity)

    print(
        f"L={L:2d} | "
        f"Relative Error={error:.4f} | "
        f"Cosine Similarity={similarity:.4f}"
    )


plt.plot(table_counts, errors, marker="o")

plt.xlabel("Number of Hash Tables (L)")
plt.ylabel("Relative Error")
plt.title("RACE Approximation Error")
plt.grid()

plt.savefig("race_accuracy.png")
plt.show()