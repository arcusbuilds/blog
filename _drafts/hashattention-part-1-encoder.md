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

For one vector, in plain C++ terms:

```cpp
int code = 0;
for (int j = 0; j < 32; j++) {          // one bit per column of W
    float dot = 0;
    for (int d = 0; d < 128; d++)       // dot product of x with column j
        dot += x[d] * W[d][j];
    if (dot > 0) code |= (1 << j);      // positive -> set bit j
}
```

Packing is just adding up the bit values where the answer is yes. With 4 bits instead of 32:

```
x @ W      = [ 0.8, -0.5, 1.1, -0.2]
> 0        = [ yes,  no,  yes,  no ]
bit values = [  1,    2,    4,    8 ]     (1 << j)
code       =    1  +  0  +  4  +  0  = 5
```

## The PyTorch reference: all vectors at once

No loops: the shapes do the work.

| Step | Operation | Shape |
|---|---|---|
| 1 | `x @ W` | (M, 128) @ (128, 32) -> (M, 32) |
| 2 | `> 0`, as 0/1 | (M, 32) |
| 3 | bit values `1 << j`, j = 0..31 | (32,) |
| 4 | multiply 2 by 3 (broadcasting lines up the columns) | (M, 32) |
| 5 | sum over the 32 columns | (M,) |
| 6 | convert to int32 | (M,) |

[paste the final `encode_ref` here once it passes]

Two details that matter:

- **Order of `@`.** A matrix multiply needs the inner sizes to match: (a, b) @ (b, c) gives (a, c). `x @ W` is
  (M, 128) @ (128, 32), which works. `W @ x` is (128, 32) @ (M, 128), which doesn't.
- **The sign bit.** Bit 31 is worth 2^31 = 2,147,483,648, one more than the largest int32. Doing the sum in
  int64 and converting to int32 at the end wraps it to the right negative number, so a code with all 32 bits
  set comes out as -1. That's correct: it's the same 32 bits.

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

- My first line was `W @ x`. Matrix multiply isn't commutative, and the shapes say so before the code even runs.
- [fill in: anything else hit while writing the reference and the kernel]

## Next

Part 2: scoring every key against the query's code with one popcount per key.
