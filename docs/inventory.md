# Qwen/Qwen3.5-0.8B — checkpoint inventory

`2fc06364715b967f1860aea9cf38778875588b17`

1 shard(s) · 488 tensors · 873,438,784 params · BF16 452, F32 36  
text path 752,393,024 (86.1%) · model card 800,000,000 (delta +73,438,784)

## Parameter budget

| bucket | class | params | MiB | % |
|---|---|---:|---:|---:|
| mtp | mtp | 20,452,864 | 39.0 | 2.3% |
| text | attention | 44,043,264 | 84.0 | 5.0% |
| text | deltanet | 189,778,752 | 362.0 | 21.7% |
| text | embedding | 254,279,680 | 485.0 | 29.1% |
| text | ffn | 264,241,152 | 504.0 | 30.3% |
| text | norm | 50,176 | 0.1 | 0.0% |
| vision | vision | 100,592,896 | 191.9 | 11.5% |
| | **TOTAL** | **873,438,784** | **1666.0** | |

## Tensors

One row per distinct tensor; `N` is the layer index.

| bucket | class | tensor | shape | dtype | count | layers | params |
|---|---|---|---|---|---:|---|---:|
| mtp | mtp | `mtp.fc.weight` | [1024, 2048] | BF16 | 1 | — | 2,097,152 |
| mtp | mtp | `mtp.layers.N.input_layernorm.weight` | [1024] | BF16 | 1 | 0 | 1,024 |
| mtp | mtp | `mtp.layers.N.mlp.down_proj.weight` | [1024, 3584] | BF16 | 1 | 0 | 3,670,016 |
| mtp | mtp | `mtp.layers.N.mlp.gate_proj.weight` | [3584, 1024] | BF16 | 1 | 0 | 3,670,016 |
| mtp | mtp | `mtp.layers.N.mlp.up_proj.weight` | [3584, 1024] | BF16 | 1 | 0 | 3,670,016 |
| mtp | mtp | `mtp.layers.N.post_attention_layernorm.weight` | [1024] | BF16 | 1 | 0 | 1,024 |
| mtp | mtp | `mtp.layers.N.self_attn.k_norm.weight` | [256] | BF16 | 1 | 0 | 256 |
| mtp | mtp | `mtp.layers.N.self_attn.k_proj.weight` | [512, 1024] | BF16 | 1 | 0 | 524,288 |
| mtp | mtp | `mtp.layers.N.self_attn.o_proj.weight` | [1024, 2048] | BF16 | 1 | 0 | 2,097,152 |
| mtp | mtp | `mtp.layers.N.self_attn.q_norm.weight` | [256] | BF16 | 1 | 0 | 256 |
| mtp | mtp | `mtp.layers.N.self_attn.q_proj.weight` | [4096, 1024] | BF16 | 1 | 0 | 4,194,304 |
| mtp | mtp | `mtp.layers.N.self_attn.v_proj.weight` | [512, 1024] | BF16 | 1 | 0 | 524,288 |
| mtp | mtp | `mtp.norm.weight` | [1024] | BF16 | 1 | — | 1,024 |
| mtp | mtp | `mtp.pre_fc_norm_embedding.weight` | [1024] | BF16 | 1 | — | 1,024 |
| mtp | mtp | `mtp.pre_fc_norm_hidden.weight` | [1024] | BF16 | 1 | — | 1,024 |
| text | attention | `model.language_model.layers.N.self_attn.k_norm.weight` | [256] | BF16 | 6 | 3, 7, 11, 15, 19, 23 | 1,536 |
| text | attention | `model.language_model.layers.N.self_attn.k_proj.weight` | [512, 1024] | BF16 | 6 | 3, 7, 11, 15, 19, 23 | 3,145,728 |
| text | attention | `model.language_model.layers.N.self_attn.o_proj.weight` | [1024, 2048] | BF16 | 6 | 3, 7, 11, 15, 19, 23 | 12,582,912 |
| text | attention | `model.language_model.layers.N.self_attn.q_norm.weight` | [256] | BF16 | 6 | 3, 7, 11, 15, 19, 23 | 1,536 |
| text | attention | `model.language_model.layers.N.self_attn.q_proj.weight` | [4096, 1024] | BF16 | 6 | 3, 7, 11, 15, 19, 23 | 25,165,824 |
| text | attention | `model.language_model.layers.N.self_attn.v_proj.weight` | [512, 1024] | BF16 | 6 | 3, 7, 11, 15, 19, 23 | 3,145,728 |
| text | deltanet | `model.language_model.layers.N.linear_attn.A_log` | [16] | F32 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 288 |
| text | deltanet | `model.language_model.layers.N.linear_attn.conv1d.weight` | [6144, 1, 4] | BF16 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 442,368 |
| text | deltanet | `model.language_model.layers.N.linear_attn.dt_bias` | [16] | BF16 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 288 |
| text | deltanet | `model.language_model.layers.N.linear_attn.in_proj_a.weight` | [16, 1024] | BF16 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 294,912 |
| text | deltanet | `model.language_model.layers.N.linear_attn.in_proj_b.weight` | [16, 1024] | BF16 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 294,912 |
| text | deltanet | `model.language_model.layers.N.linear_attn.in_proj_qkv.weight` | [6144, 1024] | BF16 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 113,246,208 |
| text | deltanet | `model.language_model.layers.N.linear_attn.in_proj_z.weight` | [2048, 1024] | BF16 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 37,748,736 |
| text | deltanet | `model.language_model.layers.N.linear_attn.norm.weight` | [128] | F32 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 2,304 |
| text | deltanet | `model.language_model.layers.N.linear_attn.out_proj.weight` | [1024, 2048] | BF16 | 18 | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22 | 37,748,736 |
| text | embedding | `model.language_model.embed_tokens.weight` | [248320, 1024] | BF16 | 1 | — | 254,279,680 |
| text | ffn | `model.language_model.layers.N.mlp.down_proj.weight` | [1024, 3584] | BF16 | 24 | 0-23 | 88,080,384 |
| text | ffn | `model.language_model.layers.N.mlp.gate_proj.weight` | [3584, 1024] | BF16 | 24 | 0-23 | 88,080,384 |
| text | ffn | `model.language_model.layers.N.mlp.up_proj.weight` | [3584, 1024] | BF16 | 24 | 0-23 | 88,080,384 |
| text | norm | `model.language_model.layers.N.input_layernorm.weight` | [1024] | BF16 | 24 | 0-23 | 24,576 |
| text | norm | `model.language_model.layers.N.post_attention_layernorm.weight` | [1024] | BF16 | 24 | 0-23 | 24,576 |
| text | norm | `model.language_model.norm.weight` | [1024] | BF16 | 1 | — | 1,024 |
| vision | vision | `model.visual.blocks.N.attn.proj.bias` | [768] | BF16 | 12 | 0-11 | 9,216 |
| vision | vision | `model.visual.blocks.N.attn.proj.weight` | [768, 768] | BF16 | 12 | 0-11 | 7,077,888 |
| vision | vision | `model.visual.blocks.N.attn.qkv.bias` | [2304] | BF16 | 12 | 0-11 | 27,648 |
| vision | vision | `model.visual.blocks.N.attn.qkv.weight` | [2304, 768] | BF16 | 12 | 0-11 | 21,233,664 |
| vision | vision | `model.visual.blocks.N.mlp.linear_fc1.bias` | [3072] | BF16 | 12 | 0-11 | 36,864 |
| vision | vision | `model.visual.blocks.N.mlp.linear_fc1.weight` | [3072, 768] | BF16 | 12 | 0-11 | 28,311,552 |
| vision | vision | `model.visual.blocks.N.mlp.linear_fc2.bias` | [768] | BF16 | 12 | 0-11 | 9,216 |
| vision | vision | `model.visual.blocks.N.mlp.linear_fc2.weight` | [768, 3072] | BF16 | 12 | 0-11 | 28,311,552 |
| vision | vision | `model.visual.blocks.N.norm1.bias` | [768] | BF16 | 12 | 0-11 | 9,216 |
| vision | vision | `model.visual.blocks.N.norm1.weight` | [768] | BF16 | 12 | 0-11 | 9,216 |
| vision | vision | `model.visual.blocks.N.norm2.bias` | [768] | BF16 | 12 | 0-11 | 9,216 |
| vision | vision | `model.visual.blocks.N.norm2.weight` | [768] | BF16 | 12 | 0-11 | 9,216 |
| vision | vision | `model.visual.merger.linear_fc1.bias` | [3072] | BF16 | 1 | — | 3,072 |
| vision | vision | `model.visual.merger.linear_fc1.weight` | [3072, 3072] | BF16 | 1 | — | 9,437,184 |
| vision | vision | `model.visual.merger.linear_fc2.bias` | [1024] | BF16 | 1 | — | 1,024 |
| vision | vision | `model.visual.merger.linear_fc2.weight` | [1024, 3072] | BF16 | 1 | — | 3,145,728 |
| vision | vision | `model.visual.merger.norm.bias` | [768] | BF16 | 1 | — | 768 |
| vision | vision | `model.visual.merger.norm.weight` | [768] | BF16 | 1 | — | 768 |
| vision | vision | `model.visual.patch_embed.proj.bias` | [768] | BF16 | 1 | — | 768 |
| vision | vision | `model.visual.patch_embed.proj.weight` | [768, 3, 2, 16, 16] | BF16 | 1 | — | 1,179,648 |
| vision | vision | `model.visual.pos_embed.weight` | [2304, 768] | BF16 | 1 | — | 1,769,472 |
