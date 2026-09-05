#!/usr/bin/env python3
import json, re, struct
from collections import Counter, defaultdict
from math import prod
from pathlib import Path
from huggingface_hub import snapshot_download

CONST = json.loads((Path(__file__).parent / "constants.json").read_text())
REPO, REVISION = CONST["repo"], CONST["revision"]
EXPECTED_PARAMS = CONST["expected_params"]

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def classify(name):
    n = name.lower()
    if "visual" in n or "vision" in n: return "vision", "vision"
    if "mtp" in n: return "mtp", "mtp"
    if "embed_tokens" in n: return "text", "embedding"
    if "lm_head" in n: return "text", "lm_head"
    if "self_attn" in n: return "text", "attention"
    if "linear_attn" in n: return "text", "deltanet"
    if "mlp" in n: return "text", "ffn"
    if "norm" in n: return "text", "norm"
    return "text", "UNCLASSIFIED"


def fmt_layers(idx):
    out, s = [], sorted(idx)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        out.append(str(s[i]) if i == j else f"{s[i]}-{s[j]}")
        i = j + 1
    return ", ".join(out)


assert fmt_layers([5, 0, 2, 1]) == "0-2, 5"


def main():
    local = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["*.safetensors", "*.json"]))
    shards = sorted(local.glob("*.safetensors"))

    tensors = []
    for shard in shards:
        for name, info in read_header(shard).items():
            numel = prod(info["shape"]) if info["shape"] else 1
            tensors.append((name, info["dtype"], info["shape"], numel,
                            numel * DTYPE_BYTES.get(info["dtype"], 0)))
    tensors.sort()

    by_sub = defaultdict(lambda: [0, 0])
    groups = defaultdict(list)  # (bucket, class, pattern, dtype, shape, numel) -> layer indices
    unclassified = []
    for name, dtype, shape, numel, nbytes in tensors:
        top, sub = classify(name)
        by_sub[(top, sub)][0] += numel
        by_sub[(top, sub)][1] += nbytes
        if sub == "UNCLASSIFIED":
            unclassified.append(name)
        idx = re.search(r"\.(\d+)\.", name)
        pattern = re.sub(r"\.\d+\.", ".N.", name)
        groups[(top, sub, pattern, dtype, tuple(shape), numel)].append(
            int(idx.group(1)) if idx else None)

    if unclassified:
        print("> **UNCLASSIFIED — fix classify() before trusting totals:**")
        for n in unclassified:
            print(f"> - `{n}`")
        print()

    total = sum(v[0] for v in by_sub.values())
    text = sum(p for (t, _), (p, _) in by_sub.items() if t == "text")
    dtypes = ", ".join(f"{d} {c}" for d, c in Counter(t[1] for t in tensors).most_common())

    print(f"# {REPO} — checkpoint inventory\n")
    print(f"`{REVISION}`\n")
    print(f"{len(shards)} shard(s) · {len(tensors)} tensors · {total:,} params · {dtypes}  ")
    print(f"text path {text:,} ({100*text/total:.1f}%)", end="")
    if EXPECTED_PARAMS:
        print(f" · model card {EXPECTED_PARAMS:,} (delta {total - EXPECTED_PARAMS:+,})", end="")
    print("\n")

    print("## Parameter budget\n")
    print("| bucket | class | params | MiB | % |")
    print("|---|---|---:|---:|---:|")
    for (top, sub), (p, b) in sorted(by_sub.items()):
        print(f"| {top} | {sub} | {p:,} | {b/2**20:.1f} | {100*p/total:.1f}% |")
    print(f"| | **TOTAL** | **{total:,}** | **{sum(v[1] for v in by_sub.values())/2**20:.1f}** | |")

    print("\n## Tensors\n")
    print("One row per distinct tensor; `N` is the layer index.\n")
    print("| bucket | class | tensor | shape | dtype | count | layers | params |")
    print("|---|---|---|---|---|---:|---|---:|")
    for (top, sub, pattern, dtype, shape, numel), layers in sorted(groups.items()):
        n = len(layers)
        where = "—" if layers[0] is None else fmt_layers(layers)
        print(f"| {top} | {sub} | `{pattern}` | {list(shape)} | {dtype} | "
              f"{n} | {where} | {numel * n:,} |")


if __name__ == "__main__":
    main()