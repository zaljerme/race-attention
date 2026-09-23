import time
import torch

from race_layer import RACEAttention
from race_layer_vectorized import VectorizedRACEAttention


torch.manual_seed(42)

dim = 128
heads = 4
L = 8
P = 4
beta = 10

old_model = RACEAttention(
    dim=dim,
    num_heads=heads,
    L=L,
    P=P,
    beta=beta
)

new_model = VectorizedRACEAttention(
    dim=dim,
    num_heads=heads,
    L=L,
    P=P,
    beta=beta
)

# Give both implementations EXACTLY the same weights/hash projections
new_model.load_state_dict(
    old_model.state_dict()
)

old_model.eval()
new_model.eval()

x = torch.randn(
    2,
    512,
    dim
)


with torch.no_grad():

    old_output = old_model(x)
    new_output = new_model(x)


max_difference = (
    old_output - new_output
).abs().max().item()

mean_difference = (
    old_output - new_output
).abs().mean().item()


print("Numerical comparison")
print("--------------------")
print(f"Max difference:  {max_difference:.10f}")
print(f"Mean difference: {mean_difference:.10f}")


def benchmark(model, runs=20):

    # warmup
    with torch.no_grad():
        for _ in range(3):
            model(x)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            model(x)

    end = time.perf_counter()

    return ((end - start) / runs) * 1000


old_time = benchmark(old_model)
new_time = benchmark(new_model)


print()
print("Performance")
print("--------------------")
print(f"Original RACE:   {old_time:.3f} ms")
print(f"Vectorized RACE: {new_time:.3f} ms")
print(
    f"Speedup:         "
    f"{old_time / new_time:.2f}x"
)