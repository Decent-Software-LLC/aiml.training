"""DECISIVE probe: is the adapter effective under HF Peft, or a no-op?

Two independent checks, run on the SAME PeftModel instance:
  (1) in-memory lora_A/lora_B norms after from_pretrained  -> did weights load?
  (2) disable_adapter_layers vs enable_adapter_layers A/B   -> does the delta apply
      at LOGIT level in the forward pass?

Run it with no warning filtering so peft "weights not used / not found"
lines are visible. Expected on a good adapter: all wrapped modules non-zero,
max|delta-logit| well above zero (ours: 11.05 on "capital of Australia").
If max|delta-logit| < 1e-5 the adapter is a no-op EVEN UNDER HF -- different
bug than the vLLM-only no-op (vLLM #49354).

Load Qwen3_5ForCausalLM explicitly -- AutoModelForCausalLM resolves Qwen3.5
to the multimodal wrapper under transformers 5.x and Peft then matches zero
modules (silent no-op, our first false negative).
"""
import torch
from transformers import Qwen3_5ForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3.5-4B"                                # or a local snapshot path
ADAPTER = "runs/q35-e2e/qwen3.5-4b-lora/final_adapter"  # from train-q35-4b-lora.py

torch.manual_seed(0)
print("Loading Qwen3_5ForCausalLM (bf16)...", flush=True)
base = Qwen3_5ForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
base.eval()

print("Attaching adapter via PeftModel.from_pretrained ...", flush=True)
lora = PeftModel.from_pretrained(base, ADAPTER)
lora.eval()
lora.print_trainable_parameters()

# (1) In-memory LoRA weight norms: find every module that has a lora_A attr
print("\n== (1) IN-MEMORY LoRA WEIGHT NORMS ==" , flush=True)
n_wrapped = 0
zero_a = zero_b = 0
samples = []
for name, m in lora.named_modules():
    if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
        n_wrapped += 1
        # iterate over active lora adapter keys
        for k in list(getattr(m, "lora_A", {}).keys()) if isinstance(getattr(m, "lora_A", {}), dict) else ["default"]:
            A = m.lora_A[k].weight if hasattr(m.lora_A[k], "weight") else m.lora_A[k]
            B = m.lora_B[k].weight if hasattr(m.lora_B[k], "weight") else m.lora_B[k]
            na, nb = A.detach().float().abs().sum().item(), B.detach().float().abs().sum().item()
            if na < 1e-8: zero_a += 1
            if nb < 1e-8: zero_b += 1
            if len(samples) < 5:
                samples.append((name, k, round(na, 3), round(nb, 3)))
print(f"  modules with lora_A & lora_B: {n_wrapped}", flush=True)
print(f"  adapters with lora_A ~0: {zero_a}   lora_B ~0: {zero_b}", flush=True)
print("  sample (path, adapter, |A|sum, |B|sum):", flush=True)
for s in samples:
    print(f"    {s}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PROMPTS = [
    "What is the capital of Australia?",
    'Translate to English: "Bonjour, comment allez-vous?"',
    "Rewrite as a haiku: the ocean at sunrise",
    'Convert to the opposite: "I love this product."',
    "Summarize in one sentence: The company reported strong quarterly earnings but warned of rising supply-chain costs.",
]
def apply_chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)

def logits_first_new_token(model, text):
    """Return the first NEW-token logits vector (post adapter) for a prompt."""
    with torch.inference_mode():
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        out = model(**inp)
        # out.logits shape [1, seq, vocab]; the next-token logits are the last position
        return out.logits[0, -1, :].float()

# (2) A/B: disable vs enable adapter on the SAME model instance, at LOGIT level
print("\n== (2) DISABLE vs ENABLE ADAPTER A/B (logit level) ===", flush=True)
lora.disable_adapter_layers()
disabled_logits = {p: logits_first_new_token(lora, apply_chat(tok, p)) for p in PROMPTS}
lora.enable_adapter_layers()
enabled_logits = {p: logits_first_new_token(lora, apply_chat(tok, p)) for p in PROMPTS}

maxdiff = 0.0
identical_logit_count = 0
for i, p in enumerate(PROMPTS):
    d = (disabled_logits[p] - enabled_logits[p]).abs()
    md = d.max().item()
    identical = bool(torch.equal(disabled_logits[p], enabled_logits[p]))
    maxdiff = max(maxdiff, md)
    if identical:
        identical_logit_count += 1
    print(f"  [{p[:44]}...] max|delta-logit|={md:.6f}  byte-identical={identical}", flush=True)

print("\n====================================", flush=True)
print(f"max |delta-logit| across prompts: {maxdiff:.6f}", flush=True)
print(f"prompts with byte-identical next-token logits: {identical_logit_count}/{len(PROMPTS)}", flush=True)
if maxdiff < 1e-5:
    print("VERDICT: LoRA delta is NOT applied in the forward pass even under HF Peft.", flush=True)
    print("  -> Weights load into the right modules but the GDN forward path bypasses them.", flush=True)
    print("  -> merge_and_unload (base_weight + B@A) does NOT go through forward, so it will still bake in the delta.", flush=True)
else:
    print("VERDICT: LoRA IS applied under HF (max|delta-logit| > 0). Adapter is effective; vLLM no-op is a serving-only path bug.", flush=True)
print("DONE_DECISIVE", flush=True)
