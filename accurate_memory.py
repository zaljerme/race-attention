import os
import time
import threading
import multiprocessing as mp

import psutil
import torch
import torch.nn as nn

from race_layer_chunked import ChunkedRACEAttention


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
            x, x, x,
            need_weights=False
        )
        return out


def worker(model_type, N, queue):

    torch.manual_seed(42)
    torch.set_num_threads(1)

    process = psutil.Process(os.getpid())

    dim = 128
    heads = 4

    if model_type == "standard":
        model = StandardAttention(
            dim=dim,
            heads=heads
        )
    else:
        model = ChunkedRACEAttention(
            dim=dim,
            num_heads=heads,
            L=8,
            P=4,
            beta=10,
            chunk_size=4
        )

    model.eval()

    x = torch.randn(1, N, dim)

    # Measure BEFORE first forward pass
    baseline = process.memory_info().rss
    peak = [baseline]

    stop = threading.Event()

    def monitor():
        while not stop.is_set():

            rss = process.memory_info().rss

            if rss > peak[0]:
                peak[0] = rss

            time.sleep(0.0005)

    thread = threading.Thread(target=monitor)
    thread.start()

    with torch.no_grad():
        output = model(x)
        _ = output.sum().item()

    stop.set()
    thread.join()

    MB = 1024 ** 2

    queue.put({
        "baseline_mb": baseline / MB,
        "peak_mb": peak[0] / MB,
        "extra_mb": (peak[0] - baseline) / MB
    })


def measure(model_type, N):

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    process = ctx.Process(
        target=worker,
        args=(model_type, N, queue)
    )

    process.start()
    process.join()

    if process.exitcode != 0:
        return None

    return queue.get()


if __name__ == "__main__":

    lengths = [
        1024,
        2048,
        4096,
        8192,
        16384
    ]

    for N in lengths:

        print()
        print("=" * 55)
        print(f"Sequence length: {N}")

        standard = measure("standard", N)
        race = measure("race", N)

        print(
            f"Standard | "
            f"Extra: {standard['extra_mb']:.1f} MB | "
            f"Peak: {standard['peak_mb']:.1f} MB"
        )

        print(
            f"RACE     | "
            f"Extra: {race['extra_mb']:.1f} MB | "
            f"Peak: {race['peak_mb']:.1f} MB"
        )

        if race["extra_mb"] > 0:
            print(
                f"Reduction: "
                f"{standard['extra_mb'] / race['extra_mb']:.2f}x"
            )