import os
import csv
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


# 1. ATTENTION RUNTIME SCALING


runtime = {}

with open("performer_benchmark.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        method = row["method"]
        n = int(row["sequence_length"])
        ms = float(row["time_ms"])

        runtime.setdefault(method, [])
        runtime[method].append((n, ms))

for method in runtime:
    runtime[method] = sorted(runtime[method])

plt.figure(figsize=(8, 5))

for method in ["Standard", "Performer", "RACE"]:
    x = [v[0] for v in runtime[method]]
    y = [v[1] for v in runtime[method]]

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        label=method
    )

plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("Sequence Length")
plt.ylabel("Median Forward Time (ms)")
plt.title("Attention Runtime Scaling")
plt.xticks(
    [1024, 2048, 4096, 8192, 16384],
    ["1K", "2K", "4K", "8K", "16K"]
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/runtime_scaling.png",
    dpi=300
)

plt.close()


# 2. RACE SPEEDUP VS STANDARD + PERFORMER


lengths = [1024, 2048, 4096, 8192, 16384]

standard = dict(runtime["Standard"])
race = dict(runtime["RACE"])
performer = dict(runtime["Performer"])

race_vs_standard = [
    standard[n] / race[n]
    for n in lengths
]

race_vs_performer = [
    performer[n] / race[n]
    for n in lengths
]

plt.figure(figsize=(8, 5))

plt.plot(
    lengths,
    race_vs_standard,
    marker="o",
    linewidth=2,
    label="RACE vs Standard"
)

plt.plot(
    lengths,
    race_vs_performer,
    marker="o",
    linewidth=2,
    label="RACE vs Performer"
)

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)

plt.xscale("log", base=2)

plt.xlabel("Sequence Length")
plt.ylabel("RACE Speedup (×)")
plt.title("RACE Attention Speedup")

plt.xticks(
    lengths,
    ["1K", "2K", "4K", "8K", "16K"]
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/race_speedup.png",
    dpi=300
)

plt.close()



# 3. MEMORY SCALING
# Fresh-process RSS measurements from accurate_memory.py


memory_lengths = [
    1024,
    2048,
    4096,
    8192,
    16384
]

standard_memory = [
    20.9,
    72.2,
    270.6,
    1050.7,
    4146.9
]

race_memory = [
    14.4,
    29.1,
    44.9,
    96.9,
    181.0
]

plt.figure(figsize=(8, 5))

plt.plot(
    memory_lengths,
    standard_memory,
    marker="o",
    linewidth=2,
    label="Standard"
)

plt.plot(
    memory_lengths,
    race_memory,
    marker="o",
    linewidth=2,
    label="RACE"
)

plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("Sequence Length")
plt.ylabel("Incremental Peak RSS (MB)")
plt.title("Attention Memory Scaling")

plt.xticks(
    memory_lengths,
    ["1K", "2K", "4K", "8K", "16K"]
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/memory_scaling.png",
    dpi=300
)

plt.close()



# 4. PIXEL CIFAR-10
# 1024-token sequences


methods = ["Standard", "RACE"]

test_accuracy = [
    26.85,
    26.30
]

training_minutes = [
    1584.46 / 60,
    2454.65 / 60
]

plt.figure(figsize=(6, 5))

bars = plt.bar(
    methods,
    test_accuracy
)

plt.ylabel("Test Accuracy (%)")
plt.title("Pixel CIFAR-10 — 1024 Tokens")
plt.ylim(0, 30)

for bar, value in zip(bars, test_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.4,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "figures/pixel_cifar_accuracy.png",
    dpi=300
)

plt.close()



# 5. SAVE FINAL SUMMARY TABLE


with open(
    "final_benchmark_summary.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Sequence Length",
        "Standard ms",
        "Performer ms",
        "RACE ms",
        "RACE vs Standard",
        "RACE vs Performer"
    ])

    for n in lengths:
        writer.writerow([
            n,
            standard[n],
            performer[n],
            race[n],
            standard[n] / race[n],
            performer[n] / race[n]
        ])


print()
print("=" * 60)
print("FINAL RESULTS GENERATED")
print("=" * 60)

print("figures/runtime_scaling.png")
print("figures/race_speedup.png")
print("figures/memory_scaling.png")
print("figures/pixel_cifar_accuracy.png")
print("final_benchmark_summary.csv")

print()

print(
    f"16K RACE vs Standard speedup: "
    f"{standard[16384] / race[16384]:.2f}x"
)

print(
    f"16K RACE vs Performer speedup: "
    f"{performer[16384] / race[16384]:.2f}x"
)

print(
    f"16K memory reduction: "
    f"{standard_memory[-1] / race_memory[-1]:.2f}x"
)