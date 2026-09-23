import argparse
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from race_layer_chunked import ChunkedRACEAttention


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class AssociativeRecallDataset(Dataset):
    def __init__(
        self,
        samples=5000,
        seq_len=256,
        num_keys=32,
        num_values=32,
        seed=42
    ):
        self.samples = samples
        self.seq_len = seq_len
        self.num_keys = num_keys
        self.num_values = num_values

        self.key_offset = 0
        self.value_offset = num_keys
        self.filler_token = num_keys + num_values
        self.vocab_size = num_keys + num_values + 1

        rng = random.Random(seed)

        self.data = []

        for _ in range(samples):

            target_key = rng.randrange(num_keys)
            target_value = rng.randrange(num_values)

            sequence = [
                self.filler_token
                for _ in range(seq_len)
            ]

            # Put correct key/value pair near beginning
            target_position = rng.randint(
                0,
                max(0, seq_len // 8)
            )

            sequence[target_position] = target_key

            sequence[target_position + 1] = (
                self.value_offset + target_value
            )

            # Distractors cannot use target key
            available_keys = [
                k
                for k in range(num_keys)
                if k != target_key
            ]

            distractor_keys = rng.sample(
                available_keys,
                8
            )

            # Prevent overlapping pairs
            used = {
                target_position,
                target_position + 1,
                seq_len - 1
            }

            for distractor_key in distractor_keys:

                while True:

                    position = rng.randint(
                        2,
                        seq_len - 3
                    )

                    if (
                        position not in used
                        and position + 1 not in used
                    ):
                        break

                distractor_value = rng.randrange(
                    num_values
                )

                sequence[position] = distractor_key

                sequence[position + 1] = (
                    self.value_offset
                    + distractor_value
                )

                used.add(position)
                used.add(position + 1)

            # Query target key at final position
            sequence[-1] = target_key

            self.data.append(
                (
                    torch.tensor(
                        sequence,
                        dtype=torch.long
                    ),
                    torch.tensor(
                        target_value,
                        dtype=torch.long
                    )
                )
            )

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return self.data[idx]


# --------------------------------------------------
# Standard Attention
# --------------------------------------------------

class StandardAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x):

        out, _ = self.attn(
            x,
            x,
            x,
            need_weights=False
        )

        return out


# --------------------------------------------------
# Transformer Block
# --------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        attention_type
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        if attention_type == "standard":

            self.attention = StandardAttention(
                dim=dim,
                heads=heads
            )

        elif attention_type == "race":

            self.attention = ChunkedRACEAttention(
                dim=dim,
                num_heads=heads,
                L=8,
                P=4,
                beta=2,
                chunk_size=4
            )

        else:

            raise ValueError(
                "attention_type must be 'standard' or 'race'"
            )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# --------------------------------------------------
# Model
# --------------------------------------------------

class RecallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        num_values,
        attention_type,
        dim=128,
        heads=4,
        layers=2
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            dim
        )

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                seq_len,
                dim
            ) * 0.02
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                attention_type=attention_type
            )
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)

        self.classifier = nn.Linear(
            dim,
            num_values
        )

    def forward(self, tokens):

        x = self.token_embedding(tokens)

        x = x + self.position_embedding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Final token contains query key
        query_representation = x[:, -1]

        return self.classifier(
            query_representation
        )


# --------------------------------------------------
# Train / Evaluate
# --------------------------------------------------

def run_experiment(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Attention:", args.attention)
    print("Sequence length:", args.seq_len)
    print("Seed:", args.seed)

    # Different but deterministic train/test sets
    train_dataset = AssociativeRecallDataset(
        samples=6000,
        seq_len=args.seq_len,
        seed=args.seed
    )

    test_dataset = AssociativeRecallDataset(
        samples=1000,
        seq_len=args.seq_len,
        seed=args.seed + 1000
    )

    # Deterministic shuffle for each seed
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = RecallTransformer(
        vocab_size=train_dataset.vocab_size,
        seq_len=args.seq_len,
        num_values=train_dataset.num_values,
        attention_type=args.attention
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )

    criterion = nn.CrossEntropyLoss()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        parameter_count
    )

    start_time = time.perf_counter()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(args.epochs):

        model.train()

        correct = 0
        total = 0
        total_loss = 0

        for tokens, labels in train_loader:

            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(tokens)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_accuracy = (
            100 * correct / total
        )

        average_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {average_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

    training_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # Testing
    # --------------------------------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for tokens, labels in test_loader:

            tokens = tokens.to(device)
            labels = labels.to(device)

            outputs = model(tokens)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    test_accuracy = (
        100 * correct / total
    )

    print()
    print("--------------------------")

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} sec"
    )

    print(
        f"Seed: {args.seed}"
    )

    print("--------------------------")


# --------------------------------------------------
# Command Line
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attention",
        choices=[
            "standard",
            "race"
        ],
        required=True
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=256
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    run_experiment(args)