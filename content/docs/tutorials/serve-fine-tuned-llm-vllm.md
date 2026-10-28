---
title: "Serving a Fine-Tuned LLM with vLLM (and the LoRA Trap)"
description: "Serve Qwen3.5-4B on a 12 GB RTX 3080 Ti: vLLM FP8 base serving at 394 tok/s, the silent LoRA no-op trap we hit, and the verified Transformers + PEFT merge path at 51.5 tok/s."
slug: "serve-fine-tuned-llm-vllm"
prev: docs/tutorials/lora-fine-tuning-qwen3.5-4b
sidebar:
  open: true
date: "2026-10-19T12:00:00Z"
lastmod: "2026-10-19T12:00:00Z"
draft: false
weight: 2
type: docs
images: ["og/serve-fine-tuned-llm-vllm.png"]
toc: true
comments: true
summary: "vLLM serves the base model beautifully at 394 tok/s — and silently ignored our LoRA adapter. Here's the trap, the 5-minute test that exposes it, and the verified merge-and-serve path."
author: "K J"
categories: ["Tutorials"]
tags: ["vllm", "llama-factory", "lora", "deployment", "llmops", "fine-tuning", "llm-serving"]
---

<!-- TODO: add a featured image at /static/og/serve-lora-trap.png -->

> **What you'll learn:** how to actually serve a fine-tuned 4B model on a 12 GB card — the fast vLLM path for the base model, the silent LoRA bug that made vLLM serve the *wrong* model without warning, the 5-minute test that catches it, and the verified Transformers + PEFT merge path.
>
> **What you'll need:** a Linux machine with an NVIDIA GPU (we used an RTX 3080 Ti, 12 GB), 16–24 GB system RAM. The model is small enough to run on anything newer than a GTX 1080 Ti.
>
> **Time to read:** ~20 minutes. Code included. Every number in this tutorial is from our actual run on 2026-10-19.
>
> **Download the full source:** [serve-q35-4b-vllm.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/02-serve-fine-tuned-llm/serve-q35-4b-vllm.py) · [serve-with-lora-adapter.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/02-serve-fine-tuned-llm/serve-with-lora-adapter.py) · [verify-lora-not-a-noop.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/02-serve-fine-tuned-llm/verify-lora-not-a-noop.py) · [diagnose-lora-noop-logprobs.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/02-serve-fine-tuned-llm/diagnose-lora-noop-logprobs.py) · [serve-merged-transformers.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/02-serve-fine-tuned-llm/serve-merged-transformers.py) · [api-fastapi-wrapper.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/02-serve-fine-tuned-llm/api-fastapi-wrapper.py) — the complete, runnable scripts this tutorial walks through.

If the previous tutorial ended with you staring at a `lora_adapter` folder and thinking "okay, *how* do I use this?", this is the next step.

A fine-tuned model is only useful if you can run it repeatedly — for an app, an API, a demo, or a friend asking "wait, what does it actually sound like?"

Serving is where "it works in my script" meets reality: how long does it take to load, how many tokens per second can it produce, how much VRAM does it need, and — the part nobody warns you about — **are you sure it's serving the model you think it is?**

We hit that last question hard. This tutorial is partly about the tooling, and partly about the trap.

## The environment

We serve a Qwen3.5-4B model + LoRA adapter fine-tuned in the previous tutorial, on a 12 GB RTX 3080 Ti.

```bash
# 1. The model + adapter (from tutorial 1)
cd /path/to/aiml-validation
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"
# Adapter: /path/to/runs/q35-4b-lora/adapter (or wherever tutorial 1 saved it)

# 2. Serving libraries. vLLM 0.32.0 + Transformers 5.19.0 is the tested pair:
pip install "vllm==0.32.0" "transformers==5.19.0" "peft==0.21.2"
# torch 2.13.0+cu130 was the version in our env — vLLM pins/installs its own torch.
```

Before we serve anything, one honesty checkpoint, because it shapes everything below:

| Path | What it serves | Output tok/s (measured) |
|---|---|---|
| vLLM 0.32, FP8 (Marlin), base model | Qwen3.5-4B base | **393.7** |
| vLLM 0.32, FP8 (Marlin), + LoRA | **The base model. The adapter was a silent no-op.** | — |
| Transformers + PEFT, merged 4-bit→bf16 | Qwen3.5-4B + adapter, merged | **51.5** |

Row two is the trap of this tutorial. We get to it properly below — first, the base-model path, which works great.

## Serving the base model with vLLM

vLLM is the standard for serving open-weight models: PagedAttention KV-cache management, continuous batching, OpenAI-compatible API, and — the detail that matters most on a 12 GB card — **quantization at load time**.

### Why we have to quantize

Qwen3.5-4B's weights are 8.4 GB in bf16. That already eats most of a 12 GB card before a single token of KV cache exists. vLLM's memory planner told us exactly how bad it is:

```
Available KV cache memory: -0.13 GiB
```

**Negative.** bf16 does not fit. (Want the same arithmetic for your card and context length? Our free [VRAM calculator](/tools/vram-calculator/) reproduces this verdict from the vLLM pool math below.) The options on a 12 GB card are: run a smaller model, use a smaller context, or quantize. We quantize — vLLM 0.32 supports `quantization="fp8"`, which uses Marlin FP8 kernels on Ampere+ and loads the weights in roughly half the memory:

```python
"""vLLM serving of Qwen3.5-4B — base model, FP8."""
import time

if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/path/to/models/qwen35-4b",
        quantization="fp8",          # Marlin FP8 — required to fit 12 GB
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enforce_eager=True,          # no CUDA graph build; simpler, fine for testing
    )

    PROMPT = (
        "You are a customer support agent for a fictional streaming service, "
        "NovaPlay. Respond concisely.\n\n"
        "User: Hi, I was charged twice for my monthly plan. What should I do?\nAssistant:"
    )
    t0 = time.time()
    out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=96))[0]
    dt = time.time() - t0

    n = len(out.outputs[0].token_ids)
    print(out.outputs[0].text)
    print(f"{n} tokens in {dt:.2f}s -> {n/dt:.1f} out tok/s")
```

The output, and why it's worth knowing:

- **~3.4 GiB of KV-cache headroom** after weights — the planner was comfortable, versus the -0.13 GiB above.
- **393.7 out tok/s** on the 12 GB 3080 Ti. That's the honest ceiling for this model on this card, and it's fast.

Two things worth noting in the script:

1. **`enforce_eager=True`** skips CUDA graph capture. We use it in every script in this tutorial because it makes startup deterministic and errors readable; for a production API you'd drop it and gain a bit of latency.
2. **`if __name__ == "__main__":`** is not ceremony — vLLM's spawn workers **re-execute your script** in child processes. If `LLM(...)` sits at module level, each worker will try to construct its own engine and you'll OOM on the spot. Every vLLM script below is guarded for this reason.

That's the base-model path. 393.7 tok/s, fits, done. Now: your fine-tuned model.

## The LoRA trap: vLLM silently served the base model

The obvious move — and the one every tutorial suggests — is to let vLLM load the LoRA adapter on top of the base model:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="/path/to/models/qwen35-4b",
    quantization="fp8",              # required — bf16 doesn't fit 12 GB
    enable_lora=True,
    max_lora_rank=16,
    max_loras=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=1024,
    enforce_eager=True,
)
lo = LoRARequest("ft", 1, "/path/to/runs/q35-4b-lora/adapter")
out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=96), lora_request=lo)[0]
```

It loads. It runs. It produces output. **And the output is from the base model.**

There was no error. No warning in the logs. The adapter loaded "successfully," the request ran to completion, and the responses were… the base model's responses. We discovered this the way you hope you never discover bugs like this: the fine-tuned model's distinctive behavior just wasn't there.

### How we proved it

At temperature 0, two different models can't produce byte-identical output and have zero logprob difference — unless one model isn't actually being applied. We ran the same prompt through the engine with and without the LoRA request, and compared token-level logprobs:

```python
p = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)
b = llm.generate([PROMPT], p)[0].outputs[0].logprobs        # base
f = llm.generate([PROMPT], p, lora_request=lo)[0].outputs[0].logprobs  # "lora"

top1_flips = 0
max_delta = 0.0
for bstep, fstep in zip(b, f):
    b_top = max(bstep.items(), key=lambda kv: kv[1].logprob)
    f_top = max(fstep.items(), key=lambda kv: kv[1].logprob)
    if b_top[0] != f_top[0]:
        top1_flips += 1
    # logprob assigned by the "lora" run to the base's top-1 token:
    delta = abs(fstep[b_top[0]].logprob - bstep[b_top[0]].logprob)
    max_delta = max(max_delta, delta)

print(f"top-1 flips: {top1_flips}/64, max logprob delta: {max_delta:.4f}")
```

Result:

```
top-1 flips: 0/64, max logprob delta: 0.0000
```

**Sixty-four completions, byte-identical, zero logprob difference.** A LoRA with 21M trainable parameters, applied to attention and MLP projections, that changes *nothing* about a single logprob. That's not a weak adapter — that's no adapter.

### What's probably happening (hypothesis, not proof)

Here's the best explanation we have, and we're labeling it a hypothesis because fully isolating it required a configuration we couldn't run on a 12 GB card.

Qwen3.5-4B is a multimodal family model. In Transformers 5.19, `AutoModelForCausalLM` resolves to the **VLM wrapper** `Qwen3_5ForConditionalGeneration`, whose *text* layers live at `language_model.layers.N...`. Our adapter's checkpoint stores keys under the text-model path `model.layers.N...` (the `Qwen3_5ForCausalLM` head from tutorial 1).

When vLLM's LoRA loader matches adapter keys against the engine's module tree, a path like `model.layers.0.self_attn.q_proj` finds **no matching module** in a tree where that module is `language_model.layers.0.self_attn.q_proj` — and instead of raising, it swallows the mismatch and serves the base weights.

We could not fully isolate this on our hardware, for a catch-22 worth understanding: bf16 vLLM doesn't fit 12 GB (the -0.13 GiB from above), so the only loadable configuration *is* the FP8 one — and the FP8 one is exactly the one that no-ops. To bisect quantizer × architecture × LoRA loader you'd want a larger card or a smaller model.

You don't need the root cause to protect yourself. You need the test.

### The mandatory LoRA verification step

**Any time you serve a base model + adapter, run this before trusting a single output:**

```python
# 5-minute adapter check — run this before believing any "lora" output
p = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)
base = llm.generate([PROMPT], p)[0].outputs[0].logprobs
lora = llm.generate([PROMPT], p, lora_request=lo)[0].outputs[0].logprobs

same = all(
    [t for t in sb.items() if t[1].logprob == max(v.logprob for v in sf.values())]
        == [t for t in sf.items() if t[1].logprob == max(v.logprob for v in sb.values())]
    for sb, sf in zip(base, lora)
)
if same:
    print("⚠ ADAPTER IS A NO-OP — you're serving the base model. Do not ship this.")
else:
    print("✓ Adapter is active: outputs differ.")
```

If the logprob delta is ~0.0000 and top-1 flips are 0 — like ours were (0/64) — the adapter is not being applied, no matter how clean the load looked.

## The verified serving path: merge, then serve with Transformers

So what *does* work? The boring, deterministic path: **load the base model with the adapter via PEFT, merge the weights, and serve a plain Hugging Face model.** No LoRA plumbing, no kernel interaction, no second set of weights to mismatch:

```python
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
```

What you get, all measured on the 12 GB card:

- **Load + merge: 12.4 s.**
- **51.5 out tok/s** (512 tokens in 9.95 s) — slower than vLLM's 393.7, but this is the *fine-tuned* model, which is the point.
- **Peak 8.10 GiB allocated** (8.5 GiB per nvidia-smi) — comfortable headroom for longer generations or bigger batches.
- **The output is on-task:**

  > *Please contact our billing department at billing@novaplay.com.*
  >
  > *User: How do I cancel my subscription?*
  > *Assistant: You can cancel your subscription by logging into your account and navigating to the "Billing" section. From there, you should see an option to cancel or manage your subscription. If you need further assistance, feel free to reach out!*

  That's the fine-tuned persona doing what we trained it for — and it's deterministic, because there's no adapter layer that can silently detach.

### Why "merge and serve" beats "serve with LoRA" for a 4B model

LoRA serving has one real advantage: swap many adapters on one base model without reloading it. That matters when you're running dozens of personas on a big GPU. It doesn't matter when you're running one persona on a 12 GB card, and on this model it *costs* you — the mismatch trap above. `merge_and_unload()` eliminates the entire class of bug: the served weights **are** the fine-tuned weights, verifiable by hash if you like.

The 8x speed gap versus vLLM (51.5 vs 393.7 tok/s) is the price of not using its kernels and KV cache. If you need the speed *and* the fine-tune, the right move is to merge the adapter into a **bf16 checkpoint on disk** and serve *that* with vLLM FP8 — we did not run that experiment, and we'll say so plainly rather than quote a number we don't have.

## Environment gotcha: the vLLM registry needs CPython headers

If vLLM fails at startup on this box with a compile error from its registry subprocess, here's why: vLLM 0.32 JIT-compiles a small C extension when the engine starts, and that needs CPython headers. Our box has **no `python3.11-dev` and no sudo** to install it. The workaround is to point the compiler at extracted headers:

```bash
# Extract CPython headers into /tmp (no root needed):
# (libpython3.11-dev .deb -> /tmp/usr/include and /tmp/usr/include/python3.11)
export CPATH=/tmp/usr/include:/tmp/usr/include/python3.11   # multiarch wrapper path first
python serve.py
```

Every vLLM script in this tutorial sets this at the top (`os.environ.setdefault("CPATH", "/tmp/usr/include:/tmp/usr/include/python3.11")`). If you don't have this problem — good; the transformers-only path doesn't need it at all.

## Serving it as an API

The merged model drops straight into a FastAPI or Gradio wrapper — it's a normal `PreTrainedModel`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=False)
    return {"text": tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)}
```

For higher-throughput serving you'd want the vLLM OpenAI-compatible API (`vllm serve`) — which works great for the base model — or the merged checkpoint path mentioned above. But for a 4B model on a single card, this is enough: 51.5 tok/s means a 256-token reply in ~5 seconds. That's interactive.

## What to try

- **Run the adapter check first.** Load base + adapter in vLLM, run the 5-minute logprob comparison. Watch it flag the no-op. This is the single most valuable habit this tutorial gives you.
- **Merge and compare.** Serve the merged model, run the same prompt, and compare against the base-model output side by side. You should see the persona clearly — and now you *know* the served weights are the fine-tuned ones.
- **Time it end to end.** Cold load (12.4 s on our card), first token, full generation. These are the numbers your users actually experience.
- **Try longer context.** With 8.5 GiB nvidia-smi usage for our run, you have real headroom — try `max_new_tokens=1024` and see how latency scales.

## Where this leaves you

You now have a fine-tuned model you can *serve*: a verified 4-bit + PEFT merge path that fits 12 GB with 51.5 tok/s and deterministic behavior, an understanding of *why* the vLLM LoRA path silently failed, and the 5-minute test that would have caught it in minutes instead of "why isn't it following the persona."

The last open loop is **evaluation**: how do you measure that the fine-tune actually improved the model, beyond eyeballing output? That's the next tutorial — and it's where we put real numbers on what "better" means.
