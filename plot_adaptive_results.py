import csv
import matplotlib.pyplot as plt

targets = []
selected_L = []
test_mean = []
test_worst = []
speedups = []

with open("adaptive_conservative_results.csv", "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        targets.append(float(row["target_cosine"]))
        selected_L.append(int(row["selected_L"]))
        test_mean.append(float(row["test_mean"]))
        test_worst.append(float(row["test_worst"]))
        speedups.append(float(row["speedup_vs_L8"]))


# ------------------------------------------------------------
# Quality vs compute budget
# ------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    targets,
    selected_L,
    marker="o",
    linewidth=2,
    label="Selected L"
)

ax1.set_xlabel("Target Cosine Similarity")
ax1.set_ylabel("Selected Hash Tables (L)")
ax1.set_xticks(targets)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    targets,
    speedups,
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Speedup vs L=8"
)

ax2.axhline(
    1.0,
    linestyle=":",
    linewidth=1
)

ax2.set_ylabel("Speedup vs Fixed L=8 (×)")

plt.title("Adaptive RACE: Quality–Latency Tradeoff")
fig.tight_layout()

plt.savefig(
    "figures/adaptive_quality_latency.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Requested vs achieved quality
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    targets,
    test_mean,
    marker="o",
    linewidth=2,
    label="Mean unseen cosine"
)

plt.plot(
    targets,
    test_worst,
    marker="s",
    linewidth=2,
    label="Worst unseen cosine"
)

plt.plot(
    targets,
    targets,
    linestyle="--",
    linewidth=1,
    label="Requested target"
)

plt.xlabel("Requested Cosine Target")
plt.ylabel("Achieved Cosine Similarity")
plt.title("Adaptive RACE Calibration Generalization")

plt.xticks(targets)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "figures/adaptive_generalization.png",
    dpi=300
)

plt.close()


print()
print("Generated:")
print("figures/adaptive_quality_latency.png")
print("figures/adaptive_generalization.png")