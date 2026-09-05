# Qwen3.5-0.8B — architecture spec

`[v]` verified against checkpoint · `[c]` verified against `config.json` · `[?]` unconfirmed

## Provenance

- `Qwen/Qwen3.5-0.8B`, revision `<sha>`
- transformers `4.57.0.dev0` `[c]` · torch / python: `<fill>`
- canary prompt: `The capital of France is Paris. The capital of Japan is`
- HF `model_type` `qwen3_5`, text `qwen3_5_text` `[c]`; inherits `qwen3_next` `[?]`

## Name prefixes `[v]`

`model.language_model.` mine · `model.visual.` skip · `mtp.` skip

## Parameter budget `[v]`

Checkpoint total 873,438,784 — matches the model card exactly.

| bucket | class | params | % of text |
|---|---|---:|---:|
| text | embedding (tied) | 254,279,680 | 33.8% |
| text | FFN | 264,241,152 | 35.1% |
| text | DeltaNet | 189,778,752 | 25.2% |
| text | attention | 44,043,264 | 5.9% |
| text | norms | 50,176 | 0.01% |
| | **text total** | **752,393,024** | |
| vision | | 100,592,896 | |
| mtp | | 20,452,864 | |

The four text classes are the phase-two attribution buckets.

## Global

| | value | |
|---|---|---|
| layers | 24 | `[v]` |
| hidden | 1024 | `[v]` |
| FFN intermediate | 3584 | `[v]` |
| vocab | 248320 | `[v]` |
| storage dtype | BF16, except `A_log` and `linear_attn.norm` in F32 | `[v]` |
| biases | none in the text path (`attention_bias: false`) | `[v]``[c]` |
| RMSNorm eps | 1e-6, one value for the whole text config | `[c]` |
| max positions | 262,144 | `[c]` |
| tied embeddings | `tie_word_embeddings: true` | `[v]``[c]` |
| MTP layers | 1 (skipped) | `[c]` |

Qwen kept the DeltaNet decay parameter and gated norm at full precision while
everything else is BF16 — a prior from the authors about where numerics are
fragile, and phase two tests it.

Internal compute is fp32 throughout phase one regardless of stored dtype.

## Layer layout `[v]`

- Attention: 3, 7, 11, 15, 19, 23 (6) — only these have `self_attn.*`
- DeltaNet: the other 18 — only these have `linear_attn.*`
- All 24 have `mlp.*`, `input_layernorm`, `post_attention_layernorm`
- `mlp_only_layers: []` — no FFN-only layers `[c]`

The config carries **both** an explicit `layer_types` array and
`full_attention_interval: 4`. Read the array; the interval is a redundant hint. `[c]`

`post_attention_layernorm` is the pre-FFN norm and exists in DeltaNet layers too.
The name is inherited, not descriptive.

## RMSNorm

Normalize before the sublayer, add the *un*normalized input back as residual. `[v]`

eps 1e-6, uniform across the text config `[c]`. (Vision blocks carry
`norm*.bias`, so those are LayerNorm, not RMSNorm — out of scope.)

Two storage conventions for the same op: `(1 + w)` with w near 0, and plain `w`
with w near 1. Loader adds 1.0 to the offset-style weights so one kernel serves
both. `[v]`

## Attention block (3, 7, 11, 15, 19, 23) `[v]`

head_dim 256, 8 query heads, 2 KV heads → 4 queries per KV head `[c]`.
Softmax scale 1/sqrt(256) — no override in config `[c]`.
`attn_output_gate: true` confirms `q_proj`'s 4096 rows are q + output gate `[c]`.

| tensor | shape | reading |
|---|---|---|
| `q_proj` | [4096, 1024] | 2048 q (8×256) + 2048 output gate |
| `k_proj` | [512, 1024] | 2 KV heads × 256 |
| `v_proj` | [512, 1024] | 2 KV heads × 256 |
| `o_proj` | [1024, 2048] | 8×256 → residual width |
| `q_norm` / `k_norm` | [256] | per head_dim, shared across heads |

RoPE `[c]`: `partial_rotary_factor 0.25` → 64 of 256 dims rotated (32 pairs),
theta 1e7, `rope_type: default`, mRoPE sections [11,11,10] (sums to 32 pairs),
`mrope_interleaved: true`.

## DeltaNet block (the other 18) `[v]`

16 key heads and 16 value heads, head_dim 128 both sides `[c]` — matches
`norm` [128] and 6144 = 3 × 16 × 128. Conv kernel 4 `[c]`.
`mamba_ssm_dtype: float32` — recurrent state stays fp32, matching the F32
`A_log` and `linear_attn.norm` in the checkpoint `[c]`.

| tensor | shape | reading |
|---|---|---|
| `in_proj_qkv` | [6144, 1024] | q, k, v fused — 3 × (16 × 128) |
| `in_proj_z` | [2048, 1024] | output gate z, 16 × 128 |
| `in_proj_a` | [16, 1024] | one scalar per head → α |
| `in_proj_b` | [16, 1024] | one scalar per head → β |
| `A_log` | [16] F32 | per-head learned decay base |
| `dt_bias` | [16] | per-head bias in the α formula |
| `conv1d` | [6144, 1, 4] | depthwise, kernel 4, over **all of q, k, v** |
| `norm` | [128] F32 | gated RMSNorm, per head_dim |
| `out_proj` | [1024, 2048] | 16×128 → residual width |

Gating is scalar per head — `in_proj_a`/`in_proj_b` are rank-16, not per-channel.
The conv channel count equals the fused qkv width, so it touches all three streams.

Recurrence:

```
r  = Sᵀk                retrieve
Δv = β(v − r)           delta correction
S  = αS + kΔvᵀ          decayed write
o  = Sᵀq / sqrt(d)      read out
```

State per layer: 16 × 128 × 128 = 262,144 floats (1 MiB fp32), constant in
sequence length. Conv state per layer: 6144 × 3 = 18,432 floats.

## FFN (all 24) `[v]`

`down(SiLU(gate(x)) * up(x))`, 1024 → 3584 → 1024. `hidden_act: silu` `[c]`.
`gate_proj` [3584,1024] · `up_proj` [3584,1024] · `down_proj` [1024,3584].

## Embedding / LM head `[v]`

`embed_tokens` [248320, 1024]. **No `lm_head.weight` — tying is real** (`tie_word_embeddings: true` `[c]`), the
loader must alias the embedding. Final norm `model.language_model.norm` [1024].

---

## Unknowns

**Norms & tolerance**
- Which norm tensors use `(1+w)` vs plain `w`. Load them all, print the means.
- Tolerance floor: max-abs and relative-Frobenius gap between HF's prefill path
  and its token-by-token decode path on the canary prompt.

**FFN**
- Confirm SiLU is on the gate branch, not the up branch.

**Attention**
- q_proj output layout: `[all q | all gate]` or interleaved per head?
- QK-norm before or after RoPE?
- Output gate activation, and where it multiplies relative to `o_proj`.
- GQA grouping order — does q head 3 pair with KV head 0 or 1?

**RoPE**
- Which 64 dims: leading block or strided?
- Pairing: adjacent (0,1)(2,3)… or split-half (i, i+32)? `mrope_interleaved: true`
  describes the mRoPE sections, not the pair layout — check the code.
- Confirm all three mRoPE sections get the same position index for text.
- Confirm the other 192 dims pass through untouched.

**DeltaNet**
- Split order inside `in_proj_qkv` — [q|k|v] or otherwise?
- Is the conv applied to fused qkv before splitting or after? Confirm causal.
- α formula over `A_log`, `dt_bias`, `in_proj_a`. Softplus and exp involved.
- β activation (sigmoid?).
- Are q and k L2-normalized? What activation on v?
- Is `norm` applied per head before `out_proj`, with z as the gate?
- Confirm no RoPE here.
- Initial S and conv state at sequence start — zeros?