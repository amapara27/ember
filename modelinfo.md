# Model Info

## Layer Indices
* Full Attention Layers: 3, 7, 11, 15, 19, 23 (6)
* Delta Net consist of the rest

## Pre-norm
* x normalized before being fed into attention, then unnormalized x added back
* most layers: out = x * rsqrt(mean(x**2) + eps) * (1.0 + w) - w stored around 0.0
* deltanet: out = x * rsqrt(mean(x**2) + eps) * w - w stored around 1.0
* same operations but different storage of w

# QK Norm
* head dim = 256
* 8 q heads, 2 kv heads
* RMS norm applied to q, k before dot products - store learned weight

* q_proj: 1024 -> 4096 - 8 q heads * 256 d_k * 2 (doubled to to emit output gate)
* k_proj: 1024 -> 512 - 2 key heads 
* v_proj: 1024 -> 512 - 2 value heads
* o_proj: 2048 -> 1024 - 2048 output to 1024 to match residual