---
title: "HashAttention from scratch, part 1: one warp, 32 votes, one 32-bit code"
author: Srijan Keshri
---

<!-- DRAFT OUTLINE. Fill the [measure] spots after the kernel works. Code lives in
     C:\Users\Srijan.Keshri\dev\learn\cuda\hashattention\encoder.py / encoder.cu -->

## The job

Part 0 promised fingerprints. This post builds the kernel that makes them: a 128-number vector in, one 32-bit
code out. It runs once for every new key, when the key enters the cache, and once per head for the query on
every generated token.

## The math is three steps

```
vector (128 floats)  ->  x W (128 x 32)  ->  32 numbers  ->  keep the signs  ->  32 bits  ->  one int
```

- Multiply by a fixed matrix `W`.
- Positive becomes 1, negative becomes 0.
- Pack the 32 bits into one int32 (bit j = column j).

Random `W` is SimHash. HashAttention learns `W` from data. The kernel doesn't care which: same multiply, same
signs. This series starts with a random `W`.

## Checking a hash function without knowing the answer

The PyTorch reference is checked by properties, not by a list of expected codes:

- With `W` = identity, bit j is just "is x[j] > 0", so hand-built vectors give known codes (0, 1, 13, -1, -2^31).
- Scaling a vector by 3 must not change its code. Signs don't care about size.
- Flipping a vector (`-x`) must flip every bit.
- Similar vectors share most bits; unrelated ones share about half. [measure: share X/32 vs Y/32 with the
  final reference]

## The CUDA idea: a warp is exactly 32 threads

One warp per vector. Lane j computes one dot product (128 multiply-adds) for column j of `W` and gets one sign.
Then `__ballot_sync` collects all 32 yes/no answers into one int in one instruction. 32 threads, 32 bits, one
code.

Memory layout detail: `W` is stored as (128, 32) row-major, so at each step of the dot product the 32 lanes read
32 neighbouring floats (one coalesced read), while all of them read the same `x[d]` (a broadcast).

## Results

[measure: kernel time for e.g. 67M keys or one decode step's queries; compare to PyTorch `x @ W` + sign + pack]
[measure: is this compute-bound or memory-bound? Nsight: DRAM % vs SM %]

## What went wrong

[fill in: bugs hit while writing it]

## Next

Part 2: scoring every key against the query's code with one popcount per key.
