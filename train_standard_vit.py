import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(42)


class PatchEmbedding(nn.Module):
    def __init__(self, dim=128, patch_size=4):
        super().__init__()

        self.proj = nn.Conv2d(
            3,
            dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class StandardAttention(nn.Module):
    def __init__(self, dim=128, heads=4):
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


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attention = StandardAttention(dim, heads)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StandardViT(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 128

        self.patch_embed = PatchEmbedding(dim)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, dim)
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, 65, dim) * 0.02
        )

        self.blocks = nn.Sequential(
            TransformerBlock(dim),
            TransformerBlock(dim)
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embed(x)

        batch = x.shape[0]

        cls = self.cls_token.expand(
            batch,
            -1,
            -1
        )

        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


model = StandardViT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

criterion = nn.CrossEntropyLoss()


print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


start_training = time.perf_counter()

for epoch in range(3):

    model.train()

    correct = 0
    total = 0
    total_loss = 0

    for batch, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if batch % 100 == 0:
            print(
                f"Epoch {epoch+1} "
                f"Batch {batch} "
                f"Loss {loss.item():.4f}"
            )

    accuracy = 100 * correct / total

    print()
    print(f"Epoch {epoch+1} complete")
    print(
        f"Loss: "
        f"{total_loss / len(train_loader):.4f}"
    )
    print(
        f"Training Accuracy: "
        f"{accuracy:.2f}%"
    )
    print()


training_time = time.perf_counter() - start_training


# Test accuracy
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


test_accuracy = 100 * correct / total


print("---------------------------")
print(f"Training Time: {training_time:.2f} sec")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("---------------------------")
