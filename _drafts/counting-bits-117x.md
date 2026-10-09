---
title: "Counting bits 117x faster: from a PyTorch loop to one CUDA instruction"
---

<!-- TODO(human): 2-4 sentences. Who you are, why you're writing GPU kernels, what this series is. -->

## The problem

Sparse attention only looks at the keys that matter. To find them cheaply, HashAttention gives every key a 32-bit
code. A key is relevant if its code looks like the query's code. "Looks like" = how many of the 32 bits match.

So for every key: XOR it with the query (each 1 bit = a mismatch), count the 1 bits, subtract from 32.

```
query  = 0000 ... 0000
key    = 0000 ... 1011
XOR    = 0000 ... 1011   -> 3 mismatches -> score 29
```

Test size: 32 attention heads x 2M keys = 67M keys, 268 MB of codes. GPU: RTX 2000 Ada Laptop (8 GB).

## Version 1: PyTorch loop (289 ms)

PyTorch has no popcount, so I counted bits one position at a time:

```python
for i in range(32):
    acc += ((q >> i) & 1) == ((k >> i) & 1)
```

Correct, and painfully slow: 289 ms.

## Version 2: lookup table (61 ms)

Count the 1 bits of every byte (0-255) once and store them in a table. Then a 32-bit number is 4 lookups instead of
32 steps:

```python
x = q[:, None] ^ k
mismatches = LUT[x & 255] + LUT[(x >> 8) & 255] + LUT[(x >> 16) & 255] + LUT[(x >> 24) & 255]
```

61 ms. 4.7x faster without writing a kernel.

## Version 3: CUDA, one instruction (2.46 ms)

The GPU has a popcount instruction, `__popc`. One thread per key:

```cuda
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i >= H * N) return;
out[i] = 32 - __popc(q[i / N] ^ k[i]);
```

2.46 ms. **117x faster than version 1, on the same GPU.**

## Why the same GPU is 117x faster

The math is tiny. The cost is memory. Every PyTorch line (`>>`, `&`, `+=`) is its own kernel that reads the whole
268 MB array and writes it back. Version 1 does that around 100 times. My kernel does it once: read each key, do
XOR + popcount in a register, write the score. This is kernel fusion, and it's the whole point of writing kernels.

## Is 2.46 ms good?

The kernel moves 537 MB (read 4 bytes + write 4 bytes per key). 537 MB / 2.46 ms = **218 GB/s**.

The GPU's limit: `nvidia-smi` says memory clock 8001 MHz. GDDR6 does 2 transfers per clock = 16 Gbps per pin,
x 128-bit bus / 8 = **256 GB/s**. So the kernel runs at **85% of the hardware limit**. There isn't much left to win
here.

## The honest part: CPU vs GPU

| Version | Time (67M keys) |
|---|---|
| CPU, PyTorch loop | 1,865 ms |
| CPU, NumPy popcount (1 core) | 199 ms |
| GPU, my CUDA kernel | 2.46 ms |

"80x faster than the CPU" sounds great, but NumPy uses 1 core of 20. A perfect multi-core CPU version is also
memory-bound: laptop DDR5 at ~60-90 GB/s means ~6-9 ms at best. The fair GPU win for this kernel is about **3x**,
the ratio of memory bandwidths. Big GPU speedups on memory-bound work usually mean a weak CPU baseline.

## Bugs I hit

- `__builtin_popcount` compiled fine in device code and returned 32 for every input. No error. Use `__popc`.
- `q[index % H]` instead of `q[index / N]`: in a row-major (H, N) array, `/ N` is the row. `% H` mixes the two sizes.
- `if (index > H * N)` instead of `>=`: one thread writes one element past the end. Tests didn't catch it.

## Next

Rung 2: Quest page scoring, my first reduction kernel.

Code: <!-- link to repo once public -->
