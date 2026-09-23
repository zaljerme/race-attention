import csv
import os
import time
import threading
import statistics
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

    # Warmup
    with torch.no_grad():
        model(x)

    baseline = process.memory_info().rss
    peak = [baseline]

    stop = threading.Event()

    def monitor():

        while not stop.is_set():

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

        output = model(x)

        # Make sure computation completes
        _ = output.sum().item()

    runtime = time.perf_counter() - start

    stop.set()
    thread.join()

    MB = 1024 ** 2

    queue.put({
        "runtime_ms": runtime * 1000,
        "extra_memory_mb":
            (peak[0] - baseline) / MB
    })


def run_once(model_type, N):

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    p = ctx.Process(
        target=worker,
        args=(model_type, N, queue)
    )

    p.start()
    p.join()

    if p.exitcode != 0:
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

    trials = 7

    results = []

    for N in sequence_lengths:

        print()
        print("=" * 60)
        print(f"Sequence length: {N}")

        model_results = {}

        for model_type in [
            "standard",
            "race"
        ]:

            runtimes = []
            memories = []

            for trial in range(trials):

                result = run_once(
                    model_type,
                    N
                )

                runtimes.append(
                    result["runtime_ms"]
                )

                memories.append(
                    result["extra_memory_mb"]
                )

                print(
                    f"{model_type:8s} "
                    f"trial {trial + 1}: "
                    f"{result['runtime_ms']:.2f} ms | "
                    f"{result['extra_memory_mb']:.1f} MB"
                )

            model_results[model_type] = {
                "runtime_median":
                    statistics.median(runtimes),

                "runtime_mean":
                    statistics.mean(runtimes),

                "runtime_std":
                    statistics.stdev(runtimes),

                "memory_median":
                    statistics.median(memories)
            }

        standard = model_results[
            "standard"
        ]

        race = model_results[
            "race"
        ]

        speedup = (
            standard["runtime_median"]
            / race["runtime_median"]
        )

        memory_reduction = (
            standard["memory_median"]
            / race["memory_median"]
        )

        print()
        print(
            f"Median speedup: "
            f"{speedup:.2f}x"
        )

        print(
            f"Median memory reduction: "
            f"{memory_reduction:.2f}x"
        )

        results.append({
            "sequence_length": N,

            "standard_runtime_ms":
                standard["runtime_median"],

            "standard_runtime_std":
                standard["runtime_std"],

            "race_runtime_ms":
                race["runtime_median"],

            "race_runtime_std":
                race["runtime_std"],

            "standard_memory_mb":
                standard["memory_median"],

            "race_memory_mb":
                race["memory_median"],

            "speedup":
                speedup,

            "memory_reduction":
                memory_reduction
        })


    with open(
        "benchmark_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


    print()
    print(
        "Saved results to "
        "benchmark_results.csv"
    )