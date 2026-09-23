import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedRACEAttention(nn.Module):
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
            torch.tensor(corners, dtype=torch.float32)
        )

        W = torch.randn(
            L,
            num_heads,
            P,
            self.head_dim
        )

        self.register_buffer("W", W)

    def soft_bucket_all(self, x):
        """
        x: [B, H, N, D]

        Returns:
        [B, L, H, N, R]
        """

        # Process ALL hash tables at once
        projected = torch.einsum(
            "bhnd,lhpd->blhnp",
            x,
            self.W
        )

        projected = torch.tanh(projected)

        logits = self.beta * torch.einsum(
            "blhnp,rp->blhnr",
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

        # All L tables simultaneously
        phi_q = self.soft_bucket_all(Q)
        phi_k = self.soft_bucket_all(K)

        # [B, L, H, R]
        bucket_mass = phi_k.sum(dim=3)

        # [B, L, H, R, D]
        bucket_values = torch.einsum(
            "blhnr,bhnd->blhrd",
            phi_k,
            V
        )

        # [B, L, H, N, D]
        numerator = torch.einsum(
            "blhnr,blhrd->blhnd",
            phi_q,
            bucket_values
        )

        # [B, L, H, N]
        denominator = torch.einsum(
            "blhnr,blhr->blhn",
            phi_q,
            bucket_mass
        )

        # Average across hash tables
        numerator = numerator.mean(dim=1)
        denominator = denominator.mean(dim=1)

        out = numerator / (
            denominator.unsqueeze(-1) + 1e-8
        )

        out = out.transpose(1, 2).reshape(
            B,
            N,
            C
        )

        return self.proj(out)