# tolerance floor: batched prefill vs token-by-token decode

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONST = json.loads((Path(__file__).parent / "constants.json").read_text())
REPO, REVISION, PROMPT = CONST["repo"], CONST["revision"], CONST["prompt"]
OUT = Path("trace") / "tolerance.json"


def main():
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    ids = tok(PROMPT, return_tensors="pt").input_ids

    with torch.no_grad():
        prefill = model(ids, use_cache=False).logits[0, -1]

        cache, decode = None, None
        for i in range(ids.shape[1]):  # one token at a time, carrying cache + DeltaNet state
            out = model(ids[:, i:i + 1], past_key_values=cache, use_cache=True)
            cache, decode = out.past_key_values, out.logits[0, -1]

    # the decode path is only meaningful if DeltaNet state actually rode along
    linear = next(L for L in cache.layers if hasattr(L, "recurrent_states"))
    conv_state = next(iter(linear.conv_states.values()))
    recurrent_state = next(iter(linear.recurrent_states.values()))
    assert recurrent_state.abs().sum() > 0, "DeltaNet state empty — decode path is not carrying it"

    diff = (prefill - decode).abs()
    tol = {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_frobenius": (torch.linalg.norm(prefill - decode) / torch.linalg.norm(prefill)).item(),
        "logit_scale_absmax": prefill.abs().max().item(),
        "argmax_agrees": (prefill.argmax() == decode.argmax()).item(),
        "top5_agrees": prefill.topk(5).indices.tolist() == decode.topk(5).indices.tolist(),
        "top1_margin": (prefill.topk(2).values[0] - prefill.topk(2).values[1]).item(),
        "prompt": PROMPT,
        "revision": REVISION,
        "compute_dtype": "float32",
        "cache_class": type(cache).__name__,
        "cache_layer_classes": sorted({type(L).__name__ for L in cache.layers}),
        "deltanet_conv_state": list(conv_state.shape),
        "deltanet_recurrent_state": list(recurrent_state.shape),
    }
    OUT.write_text(json.dumps(tol, indent=2) + "\n")
    for k, v in tol.items():
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
