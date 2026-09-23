import itertools
import math
import torch
import torch.nn.functional as F


def make_corners(P, device):
    """
    Creates every corner of a P-dimensional hypercube.

    P = 3 produces:
    [-1, -1, -1]
    [-1, -1,  1]
    ...
    [ 1,  1,  1]

    Number of buckets = 2^P
    """
    corners = list(itertools.product([-1.0, 1.0], repeat=P))
    return torch.tensor(corners, device=device)


def soft_bucket(x, W, corners, beta):
    """
    Soft LSH bucket assignment from the RACE paper.
    """

    # Random projection
    projected = torch.tanh(x @ W.T)

    # Compare projected vector to every hypercube corner
    logits = beta * (projected @ corners.T)

    # Soft assignment to buckets
    return F.softmax(logits, dim=-1)


def race_attention(Q, K, V, L=4, P=4, beta=5.0):
    """
    Educational implementation of non-causal RACE Attention.

    Q, K, V: [N, d]
    L: number of hash tables
    P: number of random hyperplanes

    Number of buckets R = 2^P
    """

    N, d = Q.shape
    device = Q.device

    corners = make_corners(P, device)

    numerator = torch.zeros_like(V)
    denominator = torch.zeros(N, device=device)

    for _ in range(L):

        # Random Gaussian hyperplanes
        W = torch.randn(P, d, device=device)

        # Soft bucket assignments
        phi_q = soft_bucket(Q, W, corners, beta)
        phi_k = soft_bucket(K, W, corners, beta)

        # Bucket mass
        A = phi_k.sum(dim=0)

        # Bucket value summaries
        B = phi_k.T @ V

        # Query the summaries
        numerator += phi_q @ B
        denominator += phi_q @ A

    numerator /= L
    denominator /= L

    output = numerator / (denominator.unsqueeze(-1) + 1e-8)

    return output


def normal_attention(Q, K, V):
    """
    Standard Transformer softmax attention.
    """

    d = Q.shape[-1]

    scores = Q @ K.T / math.sqrt(d)

    weights = F.softmax(scores, dim=-1)

    return weights @ V


if __name__ == "__main__":

    torch.manual_seed(42)

    N = 512
    d = 64

    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    softmax_output = normal_attention(Q, K, V)

    race_output = race_attention(
        Q,
        K,
        V,
        L=4,
        P=4,
        beta=5.0
    )

    print("Softmax output:", softmax_output.shape)
    print("RACE output:   ", race_output.shape)

    similarity = F.cosine_similarity(
        softmax_output.flatten(),
        race_output.flatten(),
        dim=0
    )

    print(f"Output similarity: {similarity.item():.4f}")