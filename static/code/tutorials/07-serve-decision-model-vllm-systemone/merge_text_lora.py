"""Merge a LoRA adapter into a Qwen3.5-family text decoder — the SAFE way
(tutorial #7, aiml.training).

See the full writeup: https://aiml.training/docs/tutorials/serve-decision-model-vllm-systemone/

Why this file exists: loading a hybrid GDN stack through AutoModelForCausalLM
can resolve to a wrapper whose module names do not match the adapter's, and
PeftModel.from_pretrained then attaches ZERO adapters while merge_and_unload()
saves a pristine un-fine-tuned model (a silent no-op, verified live: 0/496
attached). This script refuses to proceed unless every expected adapter tensor
is attached, and reports the merged checkpoint's declared architecture and
whether it even has an lm_head.

Usage:
  python merge_text_lora.py --base BASE_DIR --adapter ADAPTER_DIR --out OUT_DIR \
      [--expected-lora 496]

--expected-lora must be computed from your run before you trust this: count
(target_modules x matching layers) x 2 (A and B matrices). For the tutorial #6
router recipe on a 32-layer 3:1 GDN/self-attn stack (linear_attn 5 targets +
MLP 3 targets on 24 GDN layers; self_attn 4 + MLP 3 on 8 attn layers):
  24*(5+3)*2 + 8*(4+3)*2 = 384 + 112 = 496.
"""
import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base checkpoint dir")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir")
    ap.add_argument("--out", required=True, help="output merged dir")
    ap.add_argument("--expected-lora", type=int, default=None,
                    help="exact number of lora_* tensors peft must attach")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    # EXPLICIT class, not AutoModelForCausalLM: on hybrid Qwen3.5 checkpoints
    # the auto resolver can pick a wrapper whose param names mismatch the
    # adapter, producing a zero-attachment (silent no-op) merge.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    m = Qwen3_5ForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda:0")
    pm = PeftModel.from_pretrained(m, args.adapter)

    n_lora = sum(1 for n, _ in pm.named_parameters() if "lora_" in n)
    print("attached lora tensors:", n_lora)
    if args.expected_lora is not None:
        assert n_lora == args.expected_lora, (
            f"MERGE WOULD BE NO-OP: {n_lora} lora params attached, "
            f"expected {args.expected_lora}. Wrong model class or "
            "target-module mismatch — stop and inspect before merging.")
    else:
        assert n_lora > 0, "MERGE WOULD BE NO-OP: zero adapter tensors attached"

    merged = pm.merge_and_unload()
    merged.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)

    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path))
    print("saved arch:", cfg.get("architectures"),
          "model_type:", cfg.get("model_type"))

    # Sanity: an lm_head must exist for generation-style readouts (logprob
    # endpoints scoring next-token logits). Decision-only checkpoints may
    # legitimately have none — know which you have before trusting such an
    # endpoint: without a head it scores off fabricated tied embeddings.
    idx_path = os.path.join(args.out, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        keys = set(json.load(open(idx_path))["weight_map"])
        heads = [k for k in keys if "lm_head" in k]
        print("lm_head tensors:", len(heads) or "NONE — first-token logit "
              "readouts on this checkpoint are fabricated by tied weights")


if __name__ == "__main__":
    main()
