import torch

sequence_lengths = [512, 1024, 2048, 4096, 8192, 16384]

P = 4
buckets = 2 ** P

print(f"{'N':>8} {'Normal Attention':>20} {'RACE Buckets':>20}")

for N in sequence_lengths:

    # float32 = 4 bytes

    # Standard attention NxN score matrix
    normal_bytes = N * N * 4

    # RACE soft bucket representation NxR
    race_bytes = N * buckets * 4

    normal_mb = normal_bytes / (1024 ** 2)
    race_mb = race_bytes / (1024 ** 2)

    print(
        f"{N:>8} "
        f"{normal_mb:>17.2f} MB "
        f"{race_mb:>17.2f} MB"
    )