---
title: "HashAttention from scratch, part 0: why a language model rereads its whole memory for every word"
author: Srijan Keshri
---

<!-- Optional: 2-3 sentences on who you are and why you're doing this series. -->

Every time a language model writes one more token, its attention layers read every key and value it has stored
for the conversation so far. Token 30,000 rereads all 29,999 before it. The math per token is small, so the GPU
spends most of that time waiting on memory.

Sparse attention tries to read only the keys that matter. This series builds one sparse attention method,
HashAttention, from scratch in CUDA: four kernels, one post each, then an end-to-end test against normal
attention. This post is the map.

## How much does the model reread?

Take Llama-3.1-8B. Its attention has 32 layers, 8 key/value heads per layer, and 128 numbers per head. Stored as
16-bit floats, each token adds this to the cache:

```
32 layers x 8 heads x 128 numbers x 2 (key and value) x 2 bytes = 131,072 bytes = 128 KB per token
```

At a 32K-token context that's 4 GB, and generating the next token reads all of it. My laptop GPU (RTX 2000 Ada,
256 GB/s) would need about 17 ms just to move those bytes, for every single token, before doing any math.
That's arithmetic from the model's shapes, not a measurement.

To be fair about it: the model's weights are 16 GB and also get read once per token, so for one conversation at
32K tokens the cache is the smaller cost. The cache wins when contexts get longer or when a server runs many
conversations at once, because each conversation has its own cache but they all share one copy of the weights.
Eight conversations at 32K tokens means 32 GB of cache against 16 GB of weights.

## The idea: most keys barely matter

Attention gives each past key a weight, and those weights are usually far from even. A small fraction of the keys
tends to collect most of the weight. Sparse attention methods bet on that: find the important keys cheaply, then
run real attention on only those.

The hard part is the word "cheaply". If finding the important keys means reading all of them in full, nothing was
saved.

## How HashAttention finds the important keys

HashAttention turns every key into a 32-bit code: a short fingerprint where similar vectors get similar bits.
The query gets a code too. Comparing two codes means counting how many of their 32 bits match, which a GPU does
in one instruction.

The saving is in what gets read. A full key for one head is 128 numbers, 256 bytes. Its code is 4 bytes. Scoring
every key by its code reads 64 times less memory than scoring it with the full vectors.

Then the method keeps the best k keys, say 1,000 out of 32,000, and runs normal attention on just those.

## The four kernels

```
 query vector -> 32-bit code      encoder           part 4
 compare with every key's code    Hamming score     part 1
 keep the best k keys             top-k             part 2
 attention over only those keys   sparse attention  part 3
 all four at Llama-3.1-8B shapes  end to end        part 5
```

| Part | Kernel | What's new in CUDA terms |
|---|---|---|
| 1 | Hamming score | one thread per key, popcount, measuring against memory bandwidth |
| 2 | Top-k | histograms with atomics, prefix sums, writing out a variable-length list |
| 3 | Sparse attention | gathering keys by index, online softmax, splitting one query across the whole GPU |
| 4 | Encoder | matrix-vector multiply, packing 32 signs into one int with a warp instruction |
| 5 | End to end | the whole decode step, timed and checked against dense attention |

Part 3 is the hardest by a wide margin. Part 4 comes last even though it runs first, because the other kernels
can be tested with random codes, and that keeps the main path moving.

## What I'll measure, and what I won't

Each part gets a PyTorch reference first, then the CUDA kernel, checked against the reference on every run, then
timings with Nsight Compute and the same "how close is this to the hardware limit" question as part 1.

Two limits up front:

- **Random weights, real shapes.** An 8 GB laptop GPU can't hold Llama-3.1-8B, so I use random tensors with its
  layer shapes. That's enough to measure speed. It says nothing about answer quality in a real model.
- **Random hash first.** HashAttention learns its hash from data. I start with a random projection, which needs no
  training, and treat learned weights as a later step. The kernel is identical either way.

Part 1, the Hamming score kernel, is already done: 117x faster than a PyTorch loop, at 94% of the GPU's memory
bandwidth.

<!-- link to part 1 once published -->
