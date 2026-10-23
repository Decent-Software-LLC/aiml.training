---
title: "End to End: Fine-Tune, Verify, Merge, and Serve a 4B LoRA Model on One 12 GB GPU"
slug: "end-to-end-lora-pipeline"
prev: docs/tutorials/fine-tune-kev-4b-decision-model
sidebar:
  open: true
description: "The whole loop in one verified run: LoRA-train Qwen3.5-4B (train loss 1.603), prove the adapter fires (max |delta-logit| = 11.05), merge around vLLM's silent LoRA no-op (#49354), and serve the result at ~433 tok/s on one RTX 3080 Ti."
date: 2026-10-22
lastmod: 2026-10-22
draft: false
weight: 5
type: docs
images: ["og/end-to-end-lora-pipeline.png"]
toc: true
author: "K J"
categories: ["Tutorials"]
tags: ["fine-tuning", "lora", "vllm", "qwen3.5-4b", "12gb", "serving", "model-verification"]
---

> **Download the full source:** [train-q35-4b-lora.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/train-q35-4b-lora.py) · [verify-adapter-effect.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/verify-adapter-effect.py) · [merge-servable.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/merge-servable.py) · [vllm-serve-merged.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/vllm-serve-merged.py) · [vllm-serve-base-control.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/vllm-serve-base-control.py) — the complete, runnable scripts this tutorial walks through.

## What you'll run

The four previous tutorials each covered one station of the pipeline: fine-tuning, serving, evaluation, and a domain decision model. This one runs the **whole loop as a single continuous experiment** on one 12 GB card, and documents everything that happened between stations — including three bugs, one of which silently served us the wrong model:

- **Fine-tune** Qwen/Qwen3.5-4B with 4-bit LoRA on 500 Dolly examples — 96 optimizer steps, final train loss **1.603**
- **Verify** the adapter actually fires: `disable_adapter_layers()` vs `enable_adapter_layers()` at logit level, **max |Δlogit| = 11.05**, zero byte-identical outputs
- **Merge** the adapter into a dense bf16 checkpoint (**82 MB adapter → 7.9 GB model**) — the workaround for a real vLLM bug
- **Serve** the merged model with vLLM FP8 on the same 12 GB GPU — KV cache 39,116 tokens, **~433 tok/s** batched output

The honest summary up front: a trained adapter and a *served* fine-tuned model are two different facts. Most of this tutorial is about proving the second one, because our own run got fooled by a silent no-op and we're documenting exactly how.

## The run at a glance

| Stage | What ran | Result |
|---|---|---|
| Train | SFTTrainer, 4-bit NF4, r=16/α=32, 500×3 epochs | 96 steps, 936.6 s, train loss 1.603 |
| Probe | HF Peft disable/enable A/B on 5 prompts | max \|Δlogit\| 11.05, 0/5 identical |
| Merge | `merge_and_unload()` → dense save | 7.9 GB bf16 checkpoint, 5/5 prompts change behavior |
| Serve | vLLM 0.32.0, FP8, 1× RTX 3080 Ti | 39,116-token KV pool, ~433 tok/s (5-prompt batch) |

Stack (identical across all stages): torch 2.13.0+cu130, transformers 5.19.0, peft 0.21.2, trl 1.16.0, bitsandbytes 0.50.2, vLLM 0.32.0, CUDA driver 595.x. Hardware: a single RTX 3080 Ti (12 GB).

## Station 1 — Fine-tune

The training script is the same recipe as [tutorial #1](../lora-fine-tuning-qwen3.5-4b), deliberately: the point of this run is the *pipeline*, so the training config stays fixed and version-pinned. 500 examples from `databricks/databricks-dolly-15k`, converted to two-turn `messages` so training goes through the model's chat template (the same rendering it will be served with — train/serve consistency is not optional), 3 epochs, effective batch 16, cosine schedule with 3 warmup steps, LR 2e-4.

One config choice is load-bearing beyond style, so it repeats here:

```python
# NOT AutoModelForCausalLM — see "the traps" below
model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
)
```

Trainable parameters, straight from the run's stdout:

```
trainable params: 32,464,896 || all params: 4,238,216,192 || trainable%: 0.7660
```

And the loss trail (eval on rows 14000–15000 of Dolly, disjoint from the 500 training rows, every 20 steps):

| step | eval loss |
|---|---|
| 20 | 1.7915 |
| 40 | **1.7770** |
| 60 | 1.7896 |
| 80 | 1.8205 |
| 96 | 1.8198 |

Final epoch train loss: **1.603**. Two honest observations. First, eval loss bottoms at step 40 and drifts *up* — on 100 eval rows from a 15k-row dataset this curve is noisy and we would not read much into 0.04 nats; it's shown because a smooth downward eval curve from a toy eval set should make you suspicious anyway. Second, wall time: this end-to-end run took **936.6 s** for the same 3 epochs that took 279 s in the tutorial #1 run. The log holds the explanation on one line:

```
[transformers] `chunk_gated_delta_rule` is falling back to its reference
PyTorch implementation because `flash-linear-attention` is not installed.
```

Qwen3.5's backbone is hybrid — 24 GatedDeltaNet linear-attention layers plus 8 full-attention layers. When the optimized fused kernels aren't present, transformers quietly substitutes its reference PyTorch implementation and things get ~3× slower. Same weights, same hyperparameters, wildly different clock time: check your launch logs for fallback lines before concluding your experiment "got worse".

## Station 2 — Prove the adapter is live

This is the station people skip, and the one that burned us. Before merging or serving anything, run [verify-adapter-effect.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/verify-adapter-effect.py). It asks two independent questions of the same `PeftModel`:

1. **Did the weights even load?** Walk every module with `lora_A`/`lora_B` and sum their absolute values. Zero-norm adapters mean `from_pretrained` matched nothing and you're holding the base model wearing an adapter-shaped hat.
2. **Does the delta reach the forward pass?** Run each probe prompt twice on the *same instance* — once with `disable_adapter_layers()`, once with `enable_adapter_layers()` — and diff the next-token logits:

```python
lora.disable_adapter_layers()
disabled = {p: first_new_token_logits(lora, render(p)) for p in PROMPTS}
lora.enable_adapter_layers()
enabled = {p: first_new_token_logits(lora, render(p)) for p in PROMPTS}
# ours: 11.05, 8.25, 7.91, 7.75, 7.95 — and 0/5 byte-identical
```

Our verdict line, verbatim from the run:

```
max |Δlogit| across prompts: 11.046875
VERDICT: LoRA IS applied under HF (max|Δlogit| > 0). Adapter is effective;
vLLM no-op is a serving-only path bug.
```

Eleven logits' worth of difference on "What is the capital of Australia?" is not subtle: the base model launches into a verbose "Thinking Process" enumeration; the adapted model answers *"The capital of Australia is Canberra."* — the flat, direct Dolly style the fine-tune teaches. If your max |Δlogit| comes back under 1e-5, stop right here: no amount of serving work downstream can rescue an adapter that doesn't fire, and the cause is almost always station 2's famous trap (below), not your training run.

## Station 3 — Merge to a dense checkpoint

With a verified adapter, fold it into the weights and save a plain dense model ([merge-servable.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/merge-servable.py)):

```python
model = Qwen3_5ForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
lora   = PeftModel.from_pretrained(model, ADAPTER)
merged = lora.merge_and_unload()          # W' = W + (α/r)·B@A, in-place into nn.Linear
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
```

Why merge instead of serving the adapter directly? Two reasons. The first is the vLLM bug — the headline of this tutorial, covered below. The second is arithmetic: the adapter is 82 MB of deltas against a 7.9 GB bf16 model (7.9 GB written, matching the base's parameter count). For a 4B model, one fat checkpoint on disk is simpler than base-plus-adapter plumbing at every serving layer — and `merge_and_unload()` produces a model object that behaves like any stock checkpoint everywhere, including places that never heard of Qwen3.5's hybrid architecture.

The same script generates greedy completions from the base and from the merged instance on five held-out prompts. All five changed character-for-character — including "Translate to English" shifting its reasoning preamble from `Source text:` to `Input:`, the fingerprint of 500 Dolly demonstrations being baked into the weights.

## Station 4 — Serve it with vLLM (on 12 GB, where nothing fits)

Two problems stand between the merged checkpoint and a vLLM engine on this card, and both have exact fixes.

**bf16 doesn't fit.** A 7.9 GB bf16 model leaves less than the ~3 GB the KV cache needs at usable context on a 12 GB card. Same trick as the base-model serving tutorial: FP8 weights via Marlin (`quantization="fp8"`). From the engine's own startup report:

```
Actual usage is 5.23 GiB for consumed memory (weights + non-torch),
1.57 GiB for peak activation, and 0.14 GiB for CUDAGraph memory.
Available KV cache memory: 3.09 GiB
GPU KV cache size: 39,116 tokens, Maximum concurrency for 2,048 tokens
per request: 19.10x
```

Five greedy 64-token probes ran at ~433 tok/s aggregate output speed (the equivalent base-model control leg hit ~543 tok/s; FP8 cost some speed, the adapter cost none you could measure above noise at this batch size).

**flashinfer crashes the sampler JIT.** On this box (no CPython headers, no sudo), vLLM 0.32's default flashinfer sampler dies during JIT compilation. One environment variable routes sampling through the native path:

```python
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
```

Set it *before* importing `vllm`. (If you hit a different JIT failure at engine init rather than sampling, that's the registry-JIT headers issue from [tutorial #2](../serve-fine-tuned-llm-vllm) — same `CPATH` workaround.)

Then load with the usual flags — `max_model_len=2048`, `gpu_memory_utilization=0.85`, `trust_remote_code=True` — and no `enable_lora` anywhere. The LoRA is already in the weights.

## Reading the comparison honestly

The last piece of the run compares three sources on the same prompts: HF base, HF merged, and vLLM dense ([vllm-serve-merged.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/05-end-to-end-lora-pipeline/vllm-serve-merged.py)). Excerpt:

| Prompt | HF base | HF merged | vLLM dense (FP8) |
|---|---|---|---|
| Capital of Australia | *"Thinking Process:\n1. **Identify the core question…**"* | *"The capital of Australia is Canberra."* | *"The capital of Australia is Canberra."* ✓ trained style |
| Translate French greeting | *"Thinking Process: … Source text:"* | *"Thinking Process: … Input:"* | *"Hello, how are you?"* |
| Opposite of "I love this product" | *(enumerated)* | *(reworded enumeration)* | *"I hate this product"* ✓ trained style |

And the scoreboard printed its own uncomfortable line:

```
VLLM_MATCH_HF_MERGED: 0/5
```

Zero exact string matches — and *that is fine*, which is the lesson. We were comparing a bf16 HF model rendered through the chat template against an FP8-quantized vLLM engine fed the raw prompt string. Quantization changes greedy token ties; the prompt rendering changes everything else. (Our base control leg shows how wild this gets: served as *raw text completions*, even the untouched base model outputs "The capital of Australia is Canberra." — the pretrained corpus is full of exactly that QA pair. A byte-match test on raw completions would have credited the FP8 engine with teaching the base model geography.)

So the reliability ladder, strongest rung first, is: **① logit-level A/B on one HF instance** (station 2 — no tokenizer/renderer/quantization confounds at all), **② behavioral comparison inside HF** with identical rendering (base vs merged on the same prompts — ours: 5/5 differ, correct direction), and only then **③ cross-engine spot-checks**, interpreted as *style must match trained behavior*, never as byte equality. If ① passes and ③ looks wrong, believe ① and audit ③'s comparison conditions.

## The traps (each one cost us real hours)

**Trap 1 — vLLM #49354: LoRA loaded, LoRA ignored.** Serving Qwen3.5-4B with `enable_lora=True` and our freshly-trained adapter produced outputs *byte-identical to the base model*. No warning, no error, engine reports the adapter loaded. Root cause sits in vLLM's handling of the hybrid GatedDeltaNet layers: the adapter loads into the module tree but the LoRA delta never reaches the linear-attention forward path. Until upstream lands the fix: **merge, then serve dense.** You cannot detect this by watching the serving logs — you can only detect it by *having* station 2's A/B on record to compare against.

**Trap 2 — `AutoModelForCausalLM` silently trains/serves the wrong class.** Our very first oracle probe reported "adapter does nothing under HF too". It looked like a broken training run for half a day. Under transformers 5.19, `AutoModelForCausalLM` resolves Qwen3.5 to the **multimodal wrapper** (`Qwen3_5ForConditionalGeneration`, vision tower and all); PEFT attaches to that tree, matches **zero** of the language-model modules, warns politely, and proceeds. Every A/B comes back byte-identical and blames the training. Always instantiate `Qwen3_5ForCausalLM` explicitly for text-only work — and if you ever see PeftModule counts of zero in `print_trainable_parameters()` after attach, that's this trap.

**Trap 3 — flashinfer sampler JIT.** Covered above; `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM. Costs nothing on machines that don't need it.

Three symptoms, three root causes in three different layers (engine attention path, HF class resolution, JIT toolchain) — and a single plausible-looking story ("my LoRA didn't learn") covering all of them. Which is why the next section is the actual deliverable.

## The verification ladder, as a checklist

Copy this into your own pipeline. Each rung catches a class of failure the rungs below it can't:

1. **After training:** `print_trainable_parameters()` shows the expected wrapped-module count (ours: 32,464,896 params / 0.766%).
2. **After adapter load, before anything else:** norm walk over `lora_A`/`lora_B` — zero-norm modules mean PEFT matched nothing (class-resolution trap).
3. **Same instance:** disable/enable A/B at logit level — max |Δlogit| must be ≫ 0. This is the ground truth for "did the fine-tune take".
4. **Post-merge, in HF:** base vs merged generations on fixed prompts with identical chat rendering — expect systematic, direction-correct differences.
5. **Post-serve:** sampled outputs must show the *trained behavior* (compare against rung 3/4 references), not byte-match them. Log the engine's KV-cache and quantization lines while you're there — they're the first thing you'll want when numbers move later.

Rungs 1–3 cost minutes and catch everything catastrophic. Rungs 4–5 cost minutes more and catch everything embarrassing.

## Version pins and reproduction

Everything above reproduces with the exact stack from the run: **torch 2.13.0+cu130 · transformers 5.19.0 · peft 0.21.2 · trl 1.16.0 · bitsandbytes 0.50.2 · vLLM 0.32.0 · driver 595.99.02**, on one RTX 3080 Ti 12 GB (CUDA 13.0 wheel line). Model: `Qwen/Qwen3.5-4B`; dataset: `databricks/databricks-dolly-15k` rows 0–500 (train) / 14000–15000 (eval). Scripts, in pipeline order: `train-q35-4b-lora.py` → `verify-adapter-effect.py` → `merge-servable.py` → `vllm-serve-merged.py` (+ `vllm-serve-base-control.py` as the control leg). Paths at the top of each script assume local snapshots — point them wherever your models live. If your training run prints the `chunk_gated_delta_rule` fallback line, install `flash-linear-attention` for 3× faster epochs, or accept the slower reference kernels like we did.

## Where this leaves you

You've now closed the loop the previous four tutorials opened: a 4B model fine-tuned on 500 examples in fifteen minutes (five with fused kernels), a hard proof the learned delta survives the trip from adapter to engine, and a dense FP8 checkpoint answering questions at hundreds of tokens per second on a card you already own — plus the three failure modes and the ladder that turns each from a lost afternoon into a thirty-second check.

The weakest link in most DIY pipelines is not training quality. It's the silence between stations. Instrument the stations — and keep the ladder next to every new serving stack you try, because the next engine may have its own quiet way of ignoring your life's work.
