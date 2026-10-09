# LinkedIn launch drafts (series: HashAttention from scratch, parts 0 + 1)

First comment on all: "Part 0 (why) and part 1 (the kernel): https://arcusbuilds.github.io/blog/"
Media: `animations/out/part1_passes_square.mp4` (1:1) for drafts A and C, `part0_cache_square.mp4` for draft B.

## Draft A: failure-first (goal: comments)

This week I tried to speed up my CUDA kernel with shared memory.

It got 0% faster.

Some context. I'm a backend engineer learning GPU kernels on the side. My first kernel scores keys for sparse attention: XOR two 32-bit codes, count the bits.

The PyTorch version took 289 ms on my laptop GPU. My CUDA version took 2.46 ms. 117x faster, same GPU.

So I reached for the textbook next step. Put the query in shared memory. Then constant memory. Three runs each in Nsight Compute.

2.46 ms. 2.46 ms. 2.45 ms.

Nsight had already told me why, and I hadn't listened. DRAM was 94% busy. The query is 128 bytes. The keys are 268 MB. I was speeding up the part that was already free.

The lesson I'm keeping: find out which limit you're hitting before you optimize. If it's memory, only moving fewer bytes helps.

This is part 1 of a series where I build HashAttention from scratch in CUDA, one kernel per post. Part 0 explains why it matters.

What's an optimization you were sure would work, and didn't?

#CUDA #GPU

## Draft B: explain the problem (goal: saves)

Every time a language model writes one more word, it rereads everything it has stored for the conversation.

Here's how much that is.

Llama-3.1-8B stores 128 KB per token for attention. At a 32,000-token context, that's 4 GB.

It rereads those 4 GB for every new token.

On my laptop GPU that's about 17 ms per token just moving bytes, before any math. (Arithmetic from the model's shapes, not a benchmark. For a single chat the weights still cost more. The cache takes over with long contexts and many users.)

The fix people are betting on is sparse attention. Most stored keys barely matter, so find the few that do and skip the rest.

The catch: you can't read every key in full just to decide which ones to skip.

HashAttention gives each key a 32-bit fingerprint. 4 bytes instead of 256. Comparing two fingerprints is one GPU instruction.

I'm building it from scratch in CUDA, 4 kernels, one post each. The first kernel runs at 94% of my GPU's memory bandwidth.

Part 0 is the map. Link in the comments.

#CUDA #LLM

## Draft C: build log with a confession (goal: comments + likes)

Earlier this week I read a paper on sparse attention.

Now I have a CUDA kernel running at 94% of my laptop GPU's memory bandwidth.

It did not go smoothly. My first version returned a score of 0 for every input. No error. No warning. It turned out `__builtin_popcount` compiles fine in GPU code and quietly returned 32 every time. CUDA wants `__popc`.

Then my CPU comparison said the GPU was 80x faster. Also wrong. The NumPy baseline used one core and counted bits wrong on negative numbers. Redone properly over 10 runs: 9.4x against 8 CPU threads, and a hard ceiling of about 4x, because both sides are waiting on memory.

What actually made it fast had nothing to do with math. The PyTorch version made around a hundred passes over 268 MB. The kernel makes one. 289 ms down to 2.46 ms.

I'm a backend engineer by day. This is my side project: building HashAttention from scratch, one kernel per post, bugs included.

Part 0 and part 1 are up. Link in the comments.

What's the most convincing wrong number you've ever measured?

#CUDA #GPU
