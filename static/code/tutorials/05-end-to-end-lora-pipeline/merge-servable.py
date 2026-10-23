"""Workaround for vLLM #49354: merge LoRA into a dense base checkpoint.

Pipeline (single 12 GB card, everything in one process):
  1. generate base outputs (greedy)
  2. PeftModel.from_pretrained + merge_and_unload()
  3. save the dense bf16 checkpoint
  4. generate merged outputs from the SAME model instance
  5. compare base vs merged -> proves the merge changed behaviour

Load the TEXT-ONLY head (Qwen3_5ForCausalLM), never AutoModelForCausalLM:
under transformers 5.19 Auto resolves Qwen3.5 to the multimodal wrapper,
Peft matches zero modules against it, and you silently serve the base model.
"""
import json
import torch
from transformers import Qwen3_5ForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3.5-4B"                                    # or a local snapshot path
ADAPTER = "runs/q35-e2e/qwen3.5-4b-lora/final_adapter"      # from train-q35-4b-lora.py
OUT = "runs/q35-e2e/merged-dense-bf16"

tok = AutoTokenizer.from_pretrained(BASE)

def apply_chat(t, text):
    return t.apply_chat_template([{"role": "user", "content": text}],
                                 tokenize=False, add_generation_prompt=True)

PROMPTS = [
    "What is the capital of Australia?",
    'Translate to English: "Bonjour, comment allez-vous?"',
    "Rewrite as a haiku: the ocean at sunrise",
    'Convert to the opposite: "I love this product"',
    "Summarize in one sentence: The company reported strong Q3 growth driven by its AI division.",
]

model = Qwen3_5ForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

def gen(m, text, n=40):
    with torch.inference_mode():
        inp = tok(apply_chat(tok, text), return_tensors="pt",
                  truncation=True, max_length=512).to(m.device)
        out = m.generate(**inp, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inp.input_ids.shape[1]:],
                          skip_special_tokens=True).strip()

print("== 1. BASE outputs (greedy) ==", flush=True)
base_out = {p: gen(model, p) for p in PROMPTS}
for p in PROMPTS:
    print(f"  B | {p[:45]:45} | {base_out[p][:80]}", flush=True)

print("\n== 2. MERGE ==", flush=True)
lora = PeftModel.from_pretrained(model, ADAPTER)
merged = lora.merge_and_unload()
print("  merged type:", type(merged).__name__, flush=True)

print("\n== 3. SAVE dense checkpoint ==", flush=True)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("  saved to", OUT, flush=True)

print("\n== 4. MERGED outputs (greedy) ==", flush=True)
merged_out = {p: gen(merged, p) for p in PROMPTS}
for p in PROMPTS:
    print(f"  M | {p[:45]:45} | {merged_out[p][:80]}", flush=True)

diffs = [p for p in PROMPTS if base_out[p] != merged_out[p]]
print(f"\n{len(diffs)}/{len(PROMPTS)} prompts differ (base vs merged)", flush=True)
with open("runs/q35-e2e/serve/merge-compare.json", "w") as f:
    json.dump({"base": base_out, "merged": merged_out,
               "n_diff": len(diffs), "diff_prompts": diffs}, f, indent=2)
print("DONE_MERGE_SERVABLE", flush=True)
