# Part 0 social posts

Video: `animations/out/part0_cache_square.mp4` (1:1) on both. Post the blog link in the first comment on LinkedIn, in the last tweet on X.

## LinkedIn

Every time a language model writes one more word, it rereads everything it has stored for the conversation.

For Llama-3.1-8B that's 128 KB per token. At a 32K-token context, that's 4 GB, reread for every single new token.

On my laptop GPU, that's about 17 ms per token just moving bytes, before any math. (Arithmetic from the model's shapes, not a benchmark. For one chat the weights still cost more; the cache takes over with long contexts and many users at once.)

Sparse attention bets that most of those stored keys barely matter. The hard part is finding the important ones without reading all of them.

HashAttention does it with 32-bit fingerprints: 4 bytes per key instead of 256, so 64x less to read.

I'm building it from scratch in CUDA, 4 kernels, one post each. Part 0 is the map. Part 2 is the scoring kernel: it ran 117x faster than PyTorch, and then shared memory made it 0% faster.

Link in the first comment.

#CUDA #GPU #LLM #MachineLearning

First comment: Part 0: <link>

## X thread

1/ Every new token an LLM writes rereads its whole KV cache.

Llama-3.1-8B: 128 KB per token. At 32K context that's 4 GB, reread for every token.

I'm building a fix from scratch in CUDA, one kernel at a time.

2/ On my laptop GPU (256 GB/s) that's ~17 ms per token just moving bytes.

Arithmetic, not a benchmark. For one chat the weights still cost more. The cache wins at long contexts and many users.

3/ Sparse attention: most keys get almost no attention weight. Find the few that matter and attend only to those.

The catch: finding them without reading them all.

4/ HashAttention gives every key a 32-bit fingerprint. 4 bytes instead of 256, so 64x less to read.

Comparing two fingerprints is one GPU instruction (popcount).

5/ The plan: encoder, Hamming score, top-k, sparse attention, then all four at Llama-3.1-8B shapes vs dense attention.

One post per kernel. Part 0, the map: <link>
