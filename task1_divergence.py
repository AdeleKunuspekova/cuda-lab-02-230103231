"""Task 1: Warp divergence microbenchmark."""
import time
import numpy as np
from numba import cuda

N = 2 ** 20
ITERS = 1000
THREADS = 256
BLOCKS = (N + THREADS - 1) // THREADS


@cuda.jit(device=True)
def path1(v):
    # 1,000 multiply-accumulate operations
    for _ in range(ITERS):
        v = v * 1.0001 + 0.0001
    return v


@cuda.jit(device=True)
def path2(v):
    # 1,000 distinct subtract-divide operations
    for _ in range(ITERS):
        v = (v - 0.0001) / 1.0001
    return v


@cuda.jit
def kernel_a_uniform(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        for _ in range(ITERS):
            v = v * 1.0001 + 0.0001
        y[idx] = v


@cuda.jit
def kernel_b_interleaved(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        if idx % 2 == 0:          # adjacent threads diverge inside every warp
            v = path1(v)
        else:
            v = path2(v)
        y[idx] = v


@cuda.jit
def kernel_c_warp_aligned(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:      # whole warp takes the same path
            v = path1(v)
        else:
            v = path2(v)
        y[idx] = v


def benchmark(kernel, d_y, trials=10):
    kernel[BLOCKS, THREADS](d_y, N)          # warm-up (also triggers JIT compile)
    cuda.synchronize()
    times = []
    for _ in range(trials):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[BLOCKS, THREADS](d_y, N)
        cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def main():
    h_y = np.ones(N, dtype=np.float32)
    d_y = cuda.to_device(h_y)                # transfer excluded from timing

    results = {}
    for name, k in [("Kernel A (Uniform)", kernel_a_uniform),
                    ("Kernel B (Interleaved Divergence)", kernel_b_interleaved),
                    ("Kernel C (Warp-Aligned)", kernel_c_warp_aligned)]:
        d_y.copy_to_device(h_y)              # reset data for each kernel
        results[name] = benchmark(k, d_y)

    base = results["Kernel A (Uniform)"]
    print("\n| Kernel | Avg Time (ms) | Slowdown vs A |")
    print("|---|---|---|")
    for name, t in results.items():
        print(f"| {name} | {t:.4f} | {t / base:.2f}x |")


if __name__ == "__main__":
    main()
