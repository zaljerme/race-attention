import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from race_layer_chunked import ChunkedRACEAttention


# Reproducibility


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Dataset


class PixelCIFARTransform:
    """
    CIFAR-10:
        RGB image: 3 x 32 x 32

    Convert to grayscale:
        1 x 32 x 32

    Flatten:
        1024 integer pixel tokens in [0, 255]
    """

    def __call__(self, img):
        img = transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        x = np.array(img, dtype=np.uint8)
        x = torch.from_numpy(x).long()
        return x.flatten()


def build_datasets(train_size, test_size, seed):
    transform = PixelCIFARTransform()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(seed)

    train_indices = torch.randperm(
        len(train_dataset),
        generator=generator
    )[:train_size]

    test_indices = torch.randperm(
        len(test_dataset),
        generator=generator
    )[:test_size]

    return (
        Subset(train_dataset, train_indices),
        Subset(test_dataset, test_indices),
    )



# Attention


class StandardAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.attn(
            x,
            x,
            x,
            need_weights=False,
        )

        return out


# Transformer Block


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attention_type,
        mlp_ratio=4,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        if attention_type == "standard":
            self.attn = StandardAttention(
                dim=dim,
                num_heads=num_heads,
            )

        elif attention_type == "race":
            self.attn = ChunkedRACEAttention(
                dim=dim,
                num_heads=num_heads,
                L=8,
                P=4,
                beta=2.0,
                chunk_size=8,
            )

        else:
            raise ValueError(
                f"Unknown attention type: {attention_type}"
            )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):

        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# Pixel Transformer


class PixelTransformer(nn.Module):
    def __init__(
        self,
        attention_type,
        dim=64,
        num_heads=4,
        num_layers=2,
        seq_len=1024,
        num_classes=10,
    ):
        super().__init__()

        # Pixel values 0-255 become learned embeddings.
        self.pixel_embedding = nn.Embedding(
            256,
            dim,
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, seq_len, dim) * 0.02
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                attention_type=attention_type,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

        self.classifier = nn.Linear(
            dim,
            num_classes,
        )

    def forward(self, x):

        x = self.pixel_embedding(x)

        x = x + self.position_embedding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global mean pooling over all 1024 pixels.
        x = x.mean(dim=1)

        return self.classifier(x)



# Evaluation

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

    return 100.0 * correct / total


# Training


def train(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("Pixel CIFAR-10 Long-Sequence Benchmark")
    print("=" * 60)

    print("Device:", device)
    print("Attention:", args.attention)
    print("Sequence length: 1024")
    print("Train examples:", args.train_size)
    print("Test examples:", args.test_size)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Seed:", args.seed)

    train_dataset, test_dataset = build_datasets(
        args.train_size,
        args.test_size,
        args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = PixelTransformer(
        attention_type=args.attention,
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", parameter_count)
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        epoch_start = time.perf_counter()

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * y.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        train_loss = (
            total_loss / total
        )

        train_accuracy = (
            100.0
            * correct
            / total
        )

        test_accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        print(
            f"Epoch {epoch:2d} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Acc: {test_accuracy:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = (
        time.perf_counter()
        - start_time
    )

    final_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print()
    print("-" * 60)
    print(
        f"Final Test Accuracy: "
        f"{final_accuracy:.2f}%"
    )
    print(
        f"Training Time: "
        f"{total_time:.2f} sec"
    )
    print("-" * 60)

    # Save results
    results_path = Path(
        "pixel_cifar_results.csv"
    )

    file_exists = results_path.exists()

    with open(
        results_path,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "attention",
                "seed",
                "train_size",
                "test_size",
                "epochs",
                "batch_size",
                "dim",
                "layers",
                "heads",
                "parameters",
                "test_accuracy",
                "training_time_sec",
            ])

        writer.writerow([
            args.attention,
            args.seed,
            args.train_size,
            args.test_size,
            args.epochs,
            args.batch_size,
            args.dim,
            args.layers,
            args.heads,
            parameter_count,
            final_accuracy,
            total_time,
        ])

    print(
        "Saved result to "
        "pixel_cifar_results.csv"
    )


# CLI

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attention",
        choices=["standard", "race"],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    train(args)