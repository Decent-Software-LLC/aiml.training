"""Tutorial #10 aiml.training: merge the OPS-ROUTE LoRA back into a full
d1-3B checkpoint directory that vLLM can serve natively.

Why hand-rolled merge math instead of PeftModel.merge_and_unload on the full
VL class: the checkpoint's text tower is a plain LFM2 (text_config), and we
trained Lfm2ForCausalLM with the verified key map (model.language_model.X ->
model.X). The LoRA targets are text-tower linears ONLY, so merging in the
text-tower namespace and copying every other tensor through untouched is
exact and auditable. Delta math: W' = W + (B @ A) * (alpha / r).

Stage gates (handbook): every adapter key must find its base tensor; count
merged modules; output dir must contain the full tensor count of the input
(707) plus identical config/tokenizer/prompt.py files.
"""
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/home/agent/aiml-validation/models/d1-3B"
ADAPTER = "/home/agent/tierA/d1-opsroute-lora"
OUT = "/home/agent/aiml-validation/models/d1-3b-opsroute"
PRE = "model.language_model."


def main():
    cfg = json.load(open(os.path.join(ADAPTER, "adapter_config.json")))
    alpha, r = cfg["lora_alpha"], cfg["r"]
    scale = alpha / r
    print(f"[stage] adapter alpha={alpha} r={r} scale={scale}")

    adapters = {}
    with safe_open(os.path.join(ADAPTER, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            adapters[k] = f.get_tensor(k)
    # peft names: base_model.model.<path>.lora_A.default.weight
    pairs = {}
    for k, v in adapters.items():
        kind = "A" if ".lora_A." in k else "B"
        path = k.split("base_model.model.", 1)[1].split(".lora_")[0]
        pairs.setdefault(path, {})[kind] = v
    print(f"[stage] adapter module paths: {len(pairs)}")
    assert len(pairs) >= 100, f"too few adapter paths ({len(pairs)}) -- wrong adapter?"

    # Namespace bridge (verified by tensor dump 2026-11-09): peft writes
    # 'base_model.model.model.<module>.lora_{A,B}.weight' -- parsed path is
    # 'model.<module>' (training-model namespace, Lfm2ForCausalLM top level).
    # Checkpoint key 'model.language_model.model.<module>.weight' -> strip PRE
    # leaves exactly 'model.<module>.weight'; drop '.weight' => pair key.
    # (Earlier bug: slicing without checking the prefix, plus double-prefixing,
    # collapsed distinct paths -- caught by this collision gate.)
    # Direction matters: pairs keys are ALREADY in training namespace
    # ('model.layers.N.conv.in_proj'). Checkpoint keys carry PRE.
    # So transform the CHECKPOINT key at lookup time and index `pairs`
    # directly -- building a dict by transforming pair keys is backwards
    # (that is what the assertion just caught).
    # SETTLED BY DUMP (2026-11-09): checkpoint text-tower keys are
    # 'model.language_model.layers.N.<mod>.weight' -- NO second 'model.'.
    # peft pair paths are 'model.layers.N.<mod>' (Lfm2ForCausalLM top level).
    # Bridge therefore: strip PRE, drop '.weight', RE-ADD 'model.'. Only
    # checkpoint keys are ever transformed; pair keys are used as written
    # (transforming pair keys too is what caused the earlier collision).
    def ck_to_path(ck):
        assert ck.startswith(PRE), f"unexpected ck namespace: {ck}"
        p = ck[len(PRE):]
        if p.endswith(".weight"):
            p = p[:-7]
        return "model." + p

    print(f"[stage] pair sample: {sorted(pairs)[:3]}")

    src_path = os.path.join(SRC, "model.safetensors")
    out_sd = {}
    merged = 0
    with safe_open(src_path, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            w = f.get_tensor(k).to(torch.bfloat16)
            if k.startswith(PRE):
                ap = pairs.get(ck_to_path(k))
                if ap is not None:
                    A, B = ap["A"].to(torch.float32), ap["B"].to(torch.float32)
                    delta = (B @ A) * scale
                    assert tuple(delta.shape) == tuple(w.shape), (
                        f"shape mismatch {k}: delta {delta.shape} vs base {w.shape}")
                    w = (w.to(torch.float32) + delta).to(torch.bfloat16)
                    merged += 1
            out_sd[k] = w
    print(f"[stage] merged {merged} tensors (expect == adapter paths {len(pairs)})")
    assert merged == len(pairs), (
        f"merged {merged} != adapter paths {len(pairs)} -- key map mismatch")
    assert len(out_sd) == len(keys), "tensor count changed"

    os.makedirs(OUT, exist_ok=True)
    save_file(out_sd, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
    # copy every non-weight file verbatim (config, tokenizer, chat template,
    # prompt.py, generation_config, preprocessor -- vLLM + our eval need them)
    for fn in os.listdir(SRC):
        if fn == "model.safetensors" or fn.endswith(".safetensors"):
            continue
        s = os.path.join(SRC, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(OUT, fn))
    print(f"[stage] wrote {OUT}: {len(out_sd)} tensors + "
          f"{len(os.listdir(OUT)) - 1} aux files")
    # sanity: config architecture unchanged
    c = json.load(open(os.path.join(OUT, "config.json")))
    print("[stage] arch:", c.get("architectures"), "| auto_map:", bool(c.get("auto_map")))


if __name__ == "__main__":
    main()
