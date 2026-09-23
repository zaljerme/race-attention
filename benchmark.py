import time
import torch
import matplotlib.pyplot as plt

from race_attention import race_attention, normal_attention


torch.manual_seed(42)

sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
d = 64

normal_times = []
race_times = []


def benchmark(func, Q, K, V, runs=3):
    # warmup
    func(Q, K, V)

    times = []

    for _ in range(runs):
        start = time.perf_counter()

        func(Q, K, V)

        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


for N in sequence_lengths:

    print(f"\nTesting sequence length: {N}")

    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    normal_time = benchmark(
        normal_attention,
        Q,
        K,
        V
    )

    race_time = benchmark(
        lambda q, k, v: race_attention(
            q, k, v,
            L=4,
            P=4,
            beta=5.0
        ),
        Q,
        K,
        V
    )

    normal_times.append(normal_time)
    race_times.append(race_time)

    print(f"Normal Attention: {normal_time:.6f} sec")
    print(f"RACE Attention:   {race_time:.6f} sec")


plt.plot(sequence_lengths, normal_times, marker="o", label="Normal Attention")
plt.plot(sequence_lengths, race_times, marker="o", label="RACE Attention")

plt.xlabel("Sequence Length")
plt.ylabel("Runtime (seconds)")
plt.title("Attention Runtime Scaling")
plt.legend()
plt.grid()

plt.savefig("attention_benchmark.png")
plt.show()