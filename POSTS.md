# 50-post plan

Each post: full version here on the blog, short version on LinkedIn (link in first comment) and X (thread, link in last tweet).
Status: `ready` = numbers/material already exist, `later` = needs the work first.

## Series 1: HashAttention from scratch (scope locked 9 Oct 2026)

Scope = HashAttention only. vAttention sampling, Quest, Double Sparsity are out (maybe a later series).

| Part | Post | Status |
|---|---|---|
| 0 | Why a language model rereads its whole memory for every word (the map) | `_posts/2026-10-09-hashattention-part-0-why.md` |
| 1 | Hash encoder: one warp, 32 votes, one code | outline `_drafts/hashattention-part-1-encoder.md`; kernel scaffold ready (encoder.py / encoder.cu) |
| 2 | Hamming score: 117x faster than PyTorch, then 0% with shared memory | `_posts/2026-10-09-hashattention-part-2-hamming.md` |
| 3 | Top-k by histogram | kernel in progress (phase 1 TODO). Include: histogram in global memory vs shared memory (vote-counting analogy: main-office board vs classroom board), both measured; race condition demo (`++` vs `atomicAdd`) |
| 4 | Sparse attention (gather + online softmax + split-K) | later |
| 5 | End to end at Llama-3.1-8B shapes vs dense attention | later |

The tables below are the wider idea pool; many rows become sections inside the series parts above.

## Rung 1: Hamming score (material ready)

| # | Post | Status |
|---|---|---|
| 1 | Counting bits 117x faster: from a PyTorch loop to one CUDA instruction | draft in `_drafts/` |
| 2 | The lookup-table trick: 289 ms -> 61 ms without writing a kernel | ready |
| 3 | `__builtin_popcount` compiled fine on the GPU and returned 32 for everything | ready |
| 4 | My GPU is 80x faster than the CPU. The honest number is 3x | ready |
| 5 | `index % H` vs `index / N`: the row-major bug every CUDA beginner writes | ready |
| 6 | My first open-source GPU contribution: a LeetGPU challenge (PR #335) | ready |
| 7 | Kernel fusion: why 1 CUDA kernel beats 100 PyTorch ops on the same GPU | ready |
| 8 | How to compute your GPU's peak bandwidth (256 GB/s from one nvidia-smi line) | ready |
| 9 | My benchmark beat the hardware limit (277 > 256 GB/s): L2 cache lies | ready |
| 10 | Benchmark noise on a laptop GPU: when is a speedup real? | ready |
| 11 | Calling CUDA from Python on Windows with just a DLL and ctypes | ready |
| 12 | Porting my CUDA kernel to Triton | later |
| 13 | torch.compile vs my hand-written kernel | later |
| 14 | From 86% to 90% of peak: vectorized int4 loads | later |
| 15 | My kernel vs HashAttention's official one | later |

## Rung 2: Quest page scoring

| # | Post | Status |
|---|---|---|
| 16 | Quest explained: skipping whole pages of the KV cache with min/max bounds | later |
| 17 | Writing the Quest reference in PyTorch | later |
| 18 | My first reduction kernel: shared memory | later |
| 19 | Warp shuffles: reductions without shared memory | later |
| 20 | Benchmark vs the official Quest kernels | later |

## Rung 3: Double Sparsity

| # | Post | Status |
|---|---|---|
| 21 | Double Sparsity explained: 16 channels out of 128 decide attention | later |
| 22 | Gather loads: reading only the channels you need | later |
| 23 | GEMV tiling for scoring | later |
| 24 | Benchmark vs DoubleSparse kernels | later |

## Rung 4: Top-k by histogram

| # | Post | Status |
|---|---|---|
| 25 | Top-k when scores only take 33 values: use a histogram, not a sort | later |
| 26 | Shared-memory histograms and atomic contention | later |
| 27 | Prefix sums with CUB BlockScan | later |
| 28 | Stream compaction: writing out the winners | later |
| 29 | Beating torch.topk | later |
| 30 | Fusing score + top-k so scores never touch memory | later |

## Rung 5: vAttention budget

| # | Post | Status |
|---|---|---|
| 31 | vAttention explained: how many samples until the error is small enough (CLT) | later |
| 32 | Welford's algorithm: mean and variance in one pass | later |
| 33 | Random numbers inside a GPU kernel (Philox) | later |
| 34 | Benchmark vs skylight's PyTorch version | later |

## Rung 6: Sparse FlashDecoding

| # | Post | Status |
|---|---|---|
| 35 | Flash-Decoding explained for people who know CUDA but not attention | later |
| 36 | Online softmax: softmax without seeing all the numbers first | later |
| 37 | Split-K and merge: using the whole GPU for one query | later |
| 38 | GQA head layout: 32 query heads, 8 KV heads | later |
| 39 | First steps with tensor cores | later |
| 40 | Sparse decode vs dense SDPA | later |

## Finish line: the whole pipeline

| # | Post | Status |
|---|---|---|
| 41 | Score -> select -> attend: one sparse decode step end to end | later |
| 42 | Llama-3.1-8B shapes at 8K / 16K / 32K context: results | later |
| 43 | How much accuracy does sparse attention lose? Error vs dense | later |

## Background + meta

| # | Post | Status |
|---|---|---|
| 44 | Setting up CUDA on Windows (vcvars, vswhere, the `.\` trap) | ready |
| 45 | `torch.utils.cpp_extension` failed with `'std': ambiguous`: why and the workaround | ready |
| 46 | Reading the vAttention paper | later (paper not read yet) |
| 47 | Sparse attention map: Quest, Double Sparsity, HashAttention, MagicPIG | ready |
| 48 | How I picked which kernels to write (the ladder) | ready |
| 49 | First look at Nsight Compute | later |
| 50 | What 50 posts of GPU kernels taught me | later |
