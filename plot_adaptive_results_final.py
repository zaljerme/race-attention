import csv
import matplotlib.pyplot as plt

targets = []
selected_L = []
speedups = []

with open("adaptive_conservative_results.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        targets.append(float(row["target_cosine"]))
        selected_L.append(int(row["selected_L"]))
        speedups.append(float(row["speedup_vs_L8"]))

fig, ax1 = plt.subplots(figsize=(8, 5))

line1 = ax1.plot(
    targets,
    selected_L,
    marker="o",
    linewidth=2,
    label="Selected hash tables (L)"
)

ax1.set_xlabel("Target Cosine Similarity")
ax1.set_ylabel("Selected Hash Tables (L)")
ax1.set_xticks(targets)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()

line2 = ax2.plot(
    targets,
    speedups,
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Speedup vs fixed L=8"
)

baseline = ax2.axhline(
    1.0,
    linestyle=":",
    linewidth=1,
    label="Fixed L=8 baseline"
)

ax2.set_ylabel("Speedup vs Fixed L=8 (×)")

lines = line1 + line2 + [baseline]
labels = [line.get_label() for line in lines]

ax1.legend(
    lines,
    labels,
    loc="upper center"
)

plt.title("Adaptive RACE: Quality–Latency Tradeoff")

fig.tight_layout()

plt.savefig(
    "figures/adaptive_quality_latency_final.png",
    dpi=300
)

plt.close()

print("Generated figures/adaptive_quality_latency_final.png")