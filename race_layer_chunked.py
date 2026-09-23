import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedRACEAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=4,
        L=8,
        P=4,
        beta=10.0,
        chunk_size=4
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.L = L
        self.P = P
        self.beta = beta
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(
            dim,
            dim * 3
        )

        self.proj = nn.Linear(
            dim,
            dim
        )

        corners = list(
            itertools.product(
                [-1.0, 1.0],
                repeat=P
            )
        )

        self.register_buffer(
            "corners",
            torch.tensor(
                corners,
                dtype=torch.float32
            )
        )

        W = torch.randn(
            L,
            num_heads,
            P,
            self.head_dim
        )

        self.register_buffer(
            "W",
            W
        )


    def soft_bucket_chunk(
        self,
        x,
        W
    ):

        # x:
        # [B, H, N, D]

        # W:
        # [C, H, P, D]

        projected = torch.einsum(
            "bhnd,chpd->bchnp",
            x,
            W
        )

        projected = torch.tanh(
            projected
        )

        logits = self.beta * torch.einsum(
            "bchnp,rp->bchnr",
            projected,
            self.corners
        )

        return F.softmax(
            logits,
            dim=-1
        )


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

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )

        Q, K, V = qkv

        Q = F.normalize(
            Q,
            dim=-1
        )

        K = F.normalize(
            K,
            dim=-1
        )


        numerator = torch.zeros_like(V)

        denominator = torch.zeros(
            B,
            self.num_heads,
            N,
            device=x.device
        )


        for start in range(
            0,
            self.L,
            self.chunk_size
        ):

            end = min(
                start + self.chunk_size,
                self.L
            )

            W_chunk = self.W[start:end]


            phi_q = self.soft_bucket_chunk(
                Q,
                W_chunk
            )

            phi_k = self.soft_bucket_chunk(
                K,
                W_chunk
            )


            bucket_mass = phi_k.sum(
                dim=3
            )


            bucket_values = torch.einsum(
                "bchnr,bhnd->bchrd",
                phi_k,
                V
            )


            chunk_numerator = torch.einsum(
                "bchnr,bchrd->bchnd",
                phi_q,
                bucket_values
            )


            chunk_denominator = torch.einsum(
                "bchnr,bchr->bchn",
                phi_q,
                bucket_mass
            )


            numerator += chunk_numerator.sum(
                dim=1
            )

            denominator += chunk_denominator.sum(
                dim=1
            )


        numerator /= self.L
        denominator /= self.L


        out = numerator / (
            denominator.unsqueeze(-1)
            + 1e-8
        )


        out = out.transpose(
            1,
            2
        ).reshape(
            B,
            N,
            C
        )

        return self.proj(out)