# reference traace
import json, platform, re
from pathlib import Path

import numpy as np
import torch
import transformers
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
from transformers import AutoModelForCausalLM, AutoTokenizer

CONST = json.loads((Path(__file__).parent / "constants.json").read_text())
REPO, REVISION = CONST["repo"], CONST["revision"]
PROMPT = CONST["prompt"]

OUT = Path("trace")
SKIP = {"", "model", "model.layers"}  # containers, nothing of their own to record


# name formatting
def trace_name(module_path):
    fixed = {"model.embed_tokens": "embed_tokens", "model.norm": "final_norm",
             "model.rotary_emb": "rotary_emb", "lm_head": "lm_head"}
    if module_path in fixed:
        return fixed[module_path]
    m = re.fullmatch(r"model\.layers\.(\d+)(?:\.(.+))?", module_path)
    if not m:
        raise ValueError(f"unmapped module path: {module_path}")
    return f"layer_{m.group(1)}" + (f".{m.group(2)}" if m.group(2) else "")


def tensors_of(out):
    return [out] if isinstance(out, torch.Tensor) else \
           [t for t in out if isinstance(t, torch.Tensor)] if isinstance(out, (tuple, list)) else []


def main():
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    ids = tok(PROMPT, return_tensors="pt").input_ids

    captured, handles, cur = {}, [], {"layer": None}

    # clone if a block's output is the same object as its last submodule's
    def keep(name, t):
        captured[name] = t.detach().to(torch.float32).clone()

    for path, module in model.named_modules():
        if path in SKIP or path.endswith(".act_fn"):
            continue
        name = trace_name(path)

        def hook(_mod, _inp, out, name=name):
            ts = tensors_of(out)
            suffixes = ([".cos", ".sin"] if name == "rotary_emb" else
                        [""] if len(ts) == 1 else [f".{i}" for i in range(len(ts))])
            for suffix, t in zip(suffixes, ts):
                keep(name + suffix, t)

        handles.append(module.register_forward_hook(hook))
        if re.fullmatch(r"model\.layers\.\d+", path):
            i = int(path.rsplit(".", 1)[1])
            handles.append(module.register_forward_pre_hook(
                lambda _m, _i, i=i: cur.__setitem__("layer", i)))

    # conv1d and RoPE are functional calls inside forward, no module hook fires
    conv_fn, rope_fn = qwen.causal_conv1d_fn, qwen.apply_rotary_pos_emb

    def traced_conv(*a, **k):
        out = conv_fn(*a, **k)
        keep(f"layer_{cur['layer']}.linear_attn.conv1d", out)
        return out

    def traced_rope(*a, **k):
        q, k_ = rope_fn(*a, **k)
        keep(f"layer_{cur['layer']}.self_attn.q_rope", q)
        keep(f"layer_{cur['layer']}.self_attn.k_rope", k_)
        return q, k_

    qwen.causal_conv1d_fn, qwen.apply_rotary_pos_emb = traced_conv, traced_rope
    try:
        with torch.no_grad():
            logits = model(ids, use_cache=False).logits
    finally:
        qwen.causal_conv1d_fn, qwen.apply_rotary_pos_emb = conv_fn, rope_fn
        for h in handles:
            h.remove()

    captured["input_ids"] = ids.to(torch.int32)  # logits are lm_head, not stored twice

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "trace.npz", **{k: v.numpy() for k, v in captured.items()})

    top = logits[0, -1].topk(5)
    meta = {
        "repo": REPO,
        "revision": REVISION,
        "prompt": PROMPT,
        "input_ids": ids[0].tolist(),
        "input_tokens": tok.convert_ids_to_tokens(ids[0].tolist()),
        "n_tokens": ids.shape[1],
        "compute_dtype": "float32",
        "checkpoint_dtype": "bfloat16",
        "device": "cpu",
        "use_cache": False,
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "model_class": type(model).__name__,
        "layer_types": model.config.layer_types,
        "tied_embeddings": model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(),
        "n_params": sum(p.numel() for p in model.parameters()),
        "next_token": {
            "ids": top.indices.tolist(),
            "tokens": tok.convert_ids_to_tokens(top.indices.tolist()),
            "logits": [round(v, 5) for v in top.values.tolist()],
        },
        "notes": {
            "logits": "lm_head is the logits tensor; no separate copy",
            "q_rope/k_rope": "q and k after RoPE, heads-first [batch, heads, seq, head_dim]",
            "conv1d": "output of the causal depthwise conv over fused qkv",
        },
        "tensors": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "mean": round(t.float().mean().item(), 6),
                "absmax": round(t.float().abs().max().item(), 6),
            }
            for name, t in sorted(captured.items())
        },
    }
    (OUT / "trace.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"{len(captured)} tensors -> {OUT/'trace.npz'} "
          f"({(OUT/'trace.npz').stat().st_size/2**20:.1f} MiB)")
    print(f"next token: {meta['next_token']['tokens'][0]!r} (id {meta['next_token']['ids'][0]})")


if __name__ == "__main__":
    main()
