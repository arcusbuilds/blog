---
title: "HashAttention from scratch, part 2: 117x faster than PyTorch, then 0% faster with shared memory"
author: Srijan Keshri
---

Sparse attention has an awkward bootstrapping problem. To skip the keys that don't matter, you first have to
score every key, on every decode step. If that scoring pass costs as much as the attention it's meant to save,
the whole idea is dead before it starts.

This post follows one small scoring kernel through three versions. The first is a plain PyTorch loop. The last is
a single CUDA instruction per key, and it runs at 94% of my GPU's memory bandwidth. After that I tried two
classic CUDA optimizations, shared memory and constant memory, and measured exactly zero gain. Working out why they
didn't help taught me more than the speedup did.

## This is part 2 of a series

I'm building HashAttention's decode step from scratch in CUDA, one kernel per post. [Part 0]({{ site.baseurl }}{% post_url 2026-10-09-hashattention-part-0-why %}) explains why sparse
attention matters and what the full plan is. The short version:

```
 part  what happens                      kernel
 1     query vector -> 32-bit code       encoder
 2     compare with every key's code     Hamming score  <- this post
 3     keep the best k keys              top-k
 4     attention over only those keys    sparse attention
 5     all four at Llama-3.1-8B shapes   end to end
```

Part 1 (coming soon) builds the encoder that makes the 32-bit codes. This post assumes the codes exist and tests with random ones.
The series stops at HashAttention. vAttention's sampling and the Quest and Double Sparsity scorers are out of
scope for now.

## The kernel: Hamming scores for HashAttention

HashAttention gives every key a learned 32-bit code. A key looks relevant when its code agrees with the query's
code in many bit positions. So the score for a key is the number of matching bits, 0 to 32:

```
query    0000 ... 0000
key      0000 ... 1011
XOR      0000 ... 1011    -> 3 bits differ -> score 32 - 3 = 29
```

Each attention head has its own query code and its own keys. The inputs are `q` with shape `(H,)` and `k` with
shape `(H, N)`, both int32, and the output is an `(H, N)` grid of scores.

**Setup for every number below:**

- GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU, 8 GB, sm_89
- OS: Windows 11, CUDA 13 toolkit, PyTorch 2.11
- Size: H = 32 heads, N = 2,097,152 keys per head, 67M keys total (268 MB of codes)
- Timing: CUDA events, 5 warm-up calls, then the average of 10 to 50 calls
- Kernel-only timing: Nsight Compute 2026.1.1

N = 2M is far longer than the 8K to 32K contexts I'm targeting. I chose it to make memory the limit, so the timings are clean. At
those context lengths the absolute times are much smaller.

## Version 1: a PyTorch loop (289 ms)

PyTorch 2.11 has no popcount operation. So the reference compares bit `i` of the query with bit `i` of each key,
one position at a time:

```python
q = q[:, None]                     # (H,) -> (H, 1) so each row uses its own query
acc = torch.zeros_like(k)
for i in range(32):
    acc += ((q >> i) & 1) == ((k >> i) & 1)
```

`q[:, None]` turns the query list into a column, and broadcasting stretches it across each row of keys.

It's correct, and on the GPU it takes **289 ms**{: .slow}. Each line inside the loop is a separate GPU kernel that reads
the full 268 MB grid and writes a result back. 32 iterations of several kernels each means roughly a hundred
full trips through memory (my count from the code, not a profiler number).

Computing the XOR once before the loop barely moved it, to 278 ms. That was the first hint that the arithmetic
didn't matter much. The number of passes over memory did.

## Version 2: a lookup table (61 ms)

The old trick from before CPUs had popcount instructions: precompute the bit count of every byte (0 to 255)
once, then count a 32-bit number as four table lookups instead of 32 bit tests.

```python
LUT = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int32, device="cuda")

x = q[:, None] ^ k
mismatches = (LUT[x & 255] + LUT[(x >> 8) & 255]
              + LUT[(x >> 16) & 255] + LUT[(x >> 24) & 255])
score = 32 - mismatches
```

**61 ms**{: .slow}, 4.7x faster than version 1, without writing a kernel. It's still about 15 separate kernels, each
making its own pass over the data, so it's a real improvement but nowhere near the hardware.

## Version 3: one CUDA thread per key (2.46 ms)

The GPU has a popcount instruction, `__popc`. One thread per key, and the whole score is one line:

```cuda
__global__ void hamming_kernel(const int* q, const int* k, int* out, int H, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= H * N) return;
    out[i] = 32 - __popc(q[i / N] ^ k[i]);
}
```

The kernel is built into a DLL with `nvcc --shared` and called from Python through `ctypes`. Measured from
Python, including that call, it takes **2.46 ms**{: .fast}: **117x**{: .fast} faster than version 1 on the same GPU.

The speedup has nothing to do with faster math. The kernel reads each key once, does XOR, popcount and subtract
in a register, and writes each score once. Version 1 made about a hundred passes over memory; this makes one.
This is called kernel fusion, and it's the main reason people write custom kernels at all.

<video src="{{ site.baseurl }}/assets/part1/passes.mp4" poster="{{ site.baseurl }}/assets/part1/passes-poster.png"
       controls muted playsinline loop preload="metadata" style="width:100%; border-radius:8px;"
       aria-label="Animation: the PyTorch loop sweeps the 268 MB array about 100 times, the lookup table about 15, the CUDA kernel once.">
</video>

Getting there took three bugs, and none of them crashed:

- **`__builtin_popcount` in device code.** It built without errors, and every score came back 0, so the
  call was returning 32 for every input. There was no launch error and no sync error either. Switching to CUDA's `__popc` fixed it.
- **`q[i % H]` instead of `q[i / N]`.** In a row-major `(H, N)` array, `i / N` is the row. `i % H` flips between
  queries from one key to the next. The small test case caught it.
- **`if (i > H * N)` instead of `>=`.** The thread with `i == H * N` writes one int past the end of `out`. The
  tests passed anyway, because that memory usually belongs to nobody. It's still an out-of-bounds write.

## How close is that to the hardware?

The kernel moves 8 bytes per key: 4 read, 4 written. 67M keys x 8 B = 537 MB, and 537 MB in 2.46 ms is
**218 GB/s**.

The limit, from `nvidia-smi`: memory clock 8001 MHz. GDDR6 transfers twice per clock, so that's 16 Gbps per pin,
times a 128-bit bus, divided by 8: **256 GB/s**. On that estimate the kernel is at 85%.

Nsight Compute gives a sharper answer, because it times the kernel alone without the Python call:

| Metric | Value |
|---|---|
| Duration | 2.22 ms |
| DRAM throughput | 94.3% of peak |
| Memory throughput | 235 GB/s |
| Compute (SM) throughput | 29% |
| Register spills | 0 |

DRAM is 94% busy and the compute units are mostly idle. Nsight's own note on the run says the workload is
"utilizing greater than 80.0% of the available compute or memory performance." By its measurement, this kernel
is memory-bound with about 6% left before the wall.

Two measurement traps showed up along the way:

- At 1M keys the same kernel reported **277 GB/s, above the 256 GB/s limit**. The data (8 MB) fit in the GPU's
  L2 cache, so it never touched DRAM. If the test data is small, you end up benchmarking the cache.
- Two runs of the same benchmark at 8.4M keys gave 0.337 ms and 0.310 ms, 8% apart. Laptop clocks move with
  heat and power. I treat differences under about 10% as noise unless repeated runs agree.

## The fair comparison: CPU vs GPU

My first CPU comparison said the GPU was 80x faster. It was one run, against single-threaded NumPy, and the
NumPy code was wrong: `np.bitwise_count` on signed integers counts the bits of the absolute value, so `-1` gives
1 instead of 32. No error, just wrong scores for every negative code. Viewing the array as `uint32` fixes it.

So I redid it properly: correct results checked against the GPU before timing, 10 runs per case, median with
min and max. CPU work is split into 1 MB chunks so temporaries stay in cache, and capped at 8 threads.

| Case (67M keys, 537 MB moved) | Median | Min | Max | GB/s | GPU faster by |
|---|---|---|---|---|---|
| GPU, CUDA kernel | **2.54 ms**{: .fast} | 2.53 ms | 2.55 ms | 212 | |
| CPU, NumPy, 1 thread | **87.4 ms**{: .slow} | 82.5 ms | 91.9 ms | 6 | 34x |
| CPU, NumPy, 8 threads | 23.8 ms | 22.9 ms | 46.9 ms | 23 | 9.4x |
| CPU, plain copy, 8 threads | 10.3 ms | 9.9 ms | 11.1 ms | 52 | 4.1x |

The last row is the one I care about. It does no math at all, just reads and writes the same 537 MB, so it's the
fastest any CPU version of this kernel could possibly be on this laptop. My CPU memory tops out at 52 GB/s. The
GPU does the whole job at 212 GB/s. **The real ceiling on the GPU's advantage here is about 4x**, the ratio of
memory bandwidths, because neither side is doing much math.

The 8-thread NumPy version sits at 9.4x because it makes three passes over each chunk and only reaches 23 GB/s.
A hand-written C++ loop could close some of that gap. It can't get past 4x.

Two other things the 10 runs showed. The GPU was steady (2.53 to 2.55 ms). The CPU wasn't: one 8-thread run
took 46.9 ms, twice the median. With a single run I could have reported almost any number I wanted.

## Then I tried shared memory and constant memory

`q` holds only 32 values, and each one is read by 2 million threads. That's the textbook case for keeping data
in fast on-chip memory, and I expected a real gain. So I wrote two more versions.

Shared memory: each block copies `q` on chip, waits at a barrier, then reads from there.

```cuda
__global__ void hamming_shared(const int* q, const int* k, int* out, int H, int N) {
    extern __shared__ int sq[];
    for (int j = threadIdx.x; j < H; j += blockDim.x) sq[j] = q[j];
    __syncthreads();

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= H * N) return;
    out[i] = 32 - __popc(sq[i / N] ^ k[i]);
}
// launch: hamming_shared<<<blocks, 256, H * sizeof(int)>>>(q, k, out, H, N);
```

Constant memory: copy `q` once from the host side, then every thread reads it from the constant cache, which
broadcasts one value to a whole warp when all its threads read the same address.

```cuda
__constant__ int cq[1024];   // fixed size at compile time; only the first H slots are used

__global__ void hamming_const(const int* k, int* out, int H, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= H * N) return;
    out[i] = 32 - __popc(cq[i / N] ^ k[i]);
}
// host: cudaMemcpyToSymbol(cq, q, H * sizeof(int), 0, cudaMemcpyDeviceToDevice);
```

All three versions produced identical outputs. Nsight Compute, three runs each, 32 x 2M keys:

| Version | Run 1 | Run 2 | Run 3 | DRAM busy |
|---|---|---|---|---|
| Global memory | 2.46 ms | 2.45 ms | 2.45 ms | 95% |
| Shared memory | 2.46 ms | 2.46 ms | 2.46 ms | 95% |
| Constant memory | 2.44 ms | 2.45 ms | 2.53 ms | 92-95% |

There was no gain at all. Every difference is inside run-to-run noise. (These runs read 2.45 ms where the earlier profile read
2.22 ms. That was a different session with different clocks, so I only compare numbers from the same run.)

The reason was already in the first profile. DRAM was 94% busy, and `q` is 128 bytes while `k` is 268 MB. After
its first read, `q` sits in cache like any other small, hot array, so nearly all DRAM traffic is keys going in
and scores going out. Moving the 128 bytes to faster
memory speeds up the part that was already free.

Shared memory pays off when a block reuses a large amount of data many times, like the tiles of a matrix
multiply or of attention. Constant memory pays off for small read-only data that all threads read at the same
address, when the normal cache can't already hold it. Neither one reduces the bytes read from DRAM, and that
was the only number that mattered here.

## What's left on the table

The kernel can only get faster by moving fewer bytes. These are predictions; I haven't measured any of them:

- **Write scores as `uint8`.** A score is 0 to 32, so it fits in one byte. That's 5 bytes per key instead of 8,
  which predicts roughly 1.4 ms, about 1.6x faster.
- **Never write the scores at all.** The next step in HashAttention picks the top-k keys from these scores. Fusing
  scoring into that kernel leaves 4 bytes per key, which predicts about 2x.
- **Vectorized loads** (`int4`, four keys per load) might take DRAM from 94% to the high 90s. Small gain at best.
- **A Triton version** and a comparison against HashAttention's own scoring kernel. Both still to do.

## Where the time went

- PyTorch loop: **289 ms**{: .slow}. About a hundred passes over memory.
- Lookup table: **61 ms**{: .slow}. About 15 passes.
- CUDA with `__popc`: **2.46 ms**{: .fast}, one pass, 94% of DRAM bandwidth according to Nsight Compute.
- Shared memory and constant memory: 0% gain, because they don't reduce bytes read from DRAM.
- CPU vs GPU: 34x against single-threaded NumPy, 9.4x against 8 threads, and a measured ceiling of about 4x set
  by memory bandwidth (52 GB/s CPU vs 212 GB/s GPU).

The lesson I'm keeping: profile first and find which limit you're at. When Nsight says DRAM is 94% busy and the
compute units are at 29%, any optimization that doesn't move fewer bytes is wasted work. I found that out by
writing two of them.

Part 3 is top-k: picking the best keys from these scores with a histogram instead of a sort, since a score can
only be 0 to 32. It's also the kernel the scores could later be fused into, so they never get written at all.

HashAttention paper: [arXiv 2412.14468](https://arxiv.org/abs/2412.14468) (Desai et al., ICML 2025).
