import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class RACEAttention(nn.Module):
    def __init__(self, dim, num_heads=4, L=32, P=4, beta=10.0):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.L = L
        self.P = P
        self.beta = beta

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        corners = list(
            itertools.product([-1.0, 1.0], repeat=P)
        )

        self.register_buffer(
            "corners",
            torch.tensor(corners)
        )

        # Fixed random hash projections
        W = torch.randn(
            L,
            num_heads,
            P,
            self.head_dim
        )

        self.register_buffer("W", W)

    def soft_bucket(self, x, W):
        # x: [B, H, N, D]
        # W: [H, P, D]

        projected = torch.einsum(
            "bhnd,hpd->bhnp",
            x,
            W
        )

        projected = torch.tanh(projected)

        logits = self.beta * torch.einsum(
            "bhnp,rp->bhnr",
            projected,
            self.corners
        )

        return F.softmax(logits, dim=-1)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        numerator = torch.zeros_like(V)

        denominator = torch.zeros(
            B,
            self.num_heads,
            N,
            device=x.device
        )

        for l in range(self.L):

            phi_q = self.soft_bucket(
                Q,
                self.W[l]
            )

            phi_k = self.soft_bucket(
                K,
                self.W[l]
            )

            # bucket weights
            A = phi_k.sum(dim=2)

            # bucket value summaries
            bucket_values = torch.einsum(
                "bhnr,bhnd->bhrd",
                phi_k,
                V
            )

            numerator += torch.einsum(
                "bhnr,bhrd->bhnd",
                phi_q,
                bucket_values
            )

            denominator += torch.einsum(
                "bhnr,bhr->bhn",
                phi_q,
                A
            )

        numerator /= self.L
        denominator /= self.L

        out = numerator / (
            denominator.unsqueeze(-1) + 1e-8
        )

        out = out.transpose(1, 2).reshape(B, N, C)

        return self.proj(out)