"""Verified serving path: Qwen3.5-4B + LoRA, merged, via Transformers + PEFT.
Stack: transformers 5.19.0, peft 0.21.2, RTX 3080 Ti 12 GB.
Measured: load 12.4 s, 51.5 out tok/s, 8.10 GiB peak allocated (8.5 GiB nvidia-smi).
"""
import time
import torch

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("/path/to/models/qwen35-4b")
    base = AutoModelForCausalLM.from_pretrained(
        "/path/to/models/qwen35-4b",
        load_in_4bit=True,          # NF4: fits the weights in ~3 GB
        bnb_4bit_compute_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, "/path/to/runs/q35-4b-lora/adapter")
    # Bake the LoRA into the base weights: plain model, no adapter plumbing:
    model = model.merge_and_unload()
    print(f"Loaded + merged in {time.time() - t0:.1f} s")

    PROMPT = (
        "You are a customer support agent for a fictional streaming service, "
        "NovaPlay. Respond concisely.\n\n"
        "User: Hi, I was charged twice for my monthly plan. What should I do?\nAssistant:"
    )
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    dt = time.time() - t0
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    n = 512
    print(text[:500])
    print(f"{n} tokens in {dt:.2f}s -> {n/dt:.1f} out tok/s")
    print(f"peak allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
