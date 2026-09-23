import os
import time
import threading
import multiprocessing as mp

import psutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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


def worker(model_type, sequence_length, queue):

    torch.manual_seed(42)

    process = psutil.Process(os.getpid())

    dim = 128
    heads = 4

    if model_type == "standard":

        model = StandardAttention(
            dim=dim,
            heads=heads
        )

    elif model_type == "race":

        model = ChunkedRACEAttention(
            dim=dim,
            num_heads=heads,
            L=8,
            P=4,
            beta=10,
            chunk_size=4
        )

    model.eval()

    x = torch.randn(
        1,
        sequence_length,
        dim
    )

    baseline = process.memory_info().rss
    peak = [baseline]

    stop_event = threading.Event()

    def monitor():

        while not stop_event.is_set():

            rss = process.memory_info().rss

            if rss > peak[0]:
                peak[0] = rss

            time.sleep(0.001)


    thread = threading.Thread(
        target=monitor
    )

    thread.start()

    start = time.perf_counter()

    with torch.no_grad():
        y = model(x)

        # Force computation to finish
        _ = y.sum().item()

    runtime = time.perf_counter() - start

    stop_event.set()
    thread.join()

    MB = 1024 ** 2

    queue.put({
        "baseline": baseline / MB,
        "peak": peak[0] / MB,
        "extra": (peak[0] - baseline) / MB,
        "runtime": runtime * 1000
    })


def measure(model_type, sequence_length):

    ctx = mp.get_context("spawn")

    queue = ctx.Queue()

    process = ctx.Process(
        target=worker,
        args=(
            model_type,
            sequence_length,
            queue
        )
    )

    process.start()
    process.join()

    if process.exitcode != 0:
        return None

    return queue.get()


if __name__ == "__main__":

    sequence_lengths = [
        1024,
        2048,
        4096,
        8192,
        16384
    ]

    standard_memory = []
    race_memory = []

    for N in sequence_lengths:

        print()
        print("=" * 55)
        print(f"Sequence length: {N}")

        standard = measure(
            "standard",
            N
        )

        race = measure(
            "race",
            N
        )

        if standard:

            print(
                f"Standard | "
                f"Peak: {standard['peak']:.1f} MB | "
                f"Forward extra: {standard['extra']:.1f} MB | "
                f"Time: {standard['runtime']:.2f} ms"
            )

            standard_memory.append(
                standard["peak"]
            )

        else:

            print("Standard failed")

            standard_memory.append(
                float("nan")
            )


        if race:

            print(
                f"RACE     | "
                f"Peak: {race['peak']:.1f} MB | "
                f"Forward extra: {race['extra']:.1f} MB | "
                f"Time: {race['runtime']:.2f} ms"
            )

            race_memory.append(
                race["peak"]
            )

        else:

            print("RACE failed")

            race_memory.append(
                float("nan")
            )


    plt.plot(
        sequence_lengths,
        standard_memory,
        marker="o",
        label="Standard Attention"
    )

    plt.plot(
        sequence_lengths,
        race_memory,
        marker="o",
        label="RACE Attention"
    )

    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Process Memory (MB)")
    plt.title("Measured Attention Memory Scaling")

    plt.xscale("log", base=2)

    plt.legend()
    plt.grid()

    plt.savefig(
        "memory_scaling.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()