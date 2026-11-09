---
title: "Trained AND Invited: Fine-Tuning Liquid d1-3B Into a Native vLLM Decision Router"
slug: "fine-tune-d1-decision-model-vllm"
prev: docs/tutorials/liquid-d1-decision-model-vllm
sidebar:
  open: true
description: "Tutorial #9 proved stock d1-3B gets invited to vLLM's /v1/systemone but scores 23/120 on our domain. This is the other half: a rank-16 LoRA on the whitelisted architecture, merged back into a native checkpoint, serving through the decision endpoint at 119/120 with label_mass 0.9999 — the only cell of the two-axis map that is both trained and invited. Plus the four assertion gates that caught namespace bugs before they could silently corrupt the merge."
date: 2026-11-09
lastmod: 2026-11-09
draft: false
weight: 10
type: docs
images: ["og/fine-tune-d1-decision-model-vllm.png"]
toc: true
---

[Tutorial #9](/docs/tutorials/liquid-d1-decision-model-vllm/) ended on an uncomfortable number. Stock `LiquidAI/d1-3B` is the decision model vLLM actually invites through `/v1/systemone` — choice, yes/no, and score questions answered natively, with honest calibration diagnostics. On a frozen 120-example benchmark from [tutorial #6](/docs/tutorials/fine-tune-lora-decision-router/), it scored **23/120** against a 12.5% chance floor. A real decision head, honestly reporting near-uniform uncertainty on a task we can solve with a rank-8 adapter.

This tutorial closes the loop: fine-tune the *whitelisted* architecture so the result is both **trained and invited**. Same box as every episode — a single RTX 3080 Ti (12 GB) — same pinned stack (vLLM 0.33 on a CUDA 13 wheel), same frozen OPS-ROUTE dataset from #6 so every number across the arc is comparable.

**The headline:** the fine-tuned, merged d1-3B serves through `/v1/systemone` on the stock endpoint with zero registration hacks and scores **119/120 = 99.2%** on the frozen test set, `label_mass` median **0.9999**, confidence median **0.9986**. One miss out of 120.

## The two-axis map, all four cells filled

Three tutorials have been building toward this table. It is the whole point of the decision-model series:

| | stock weights | fine-tuned (LoRA) |
|---|---|---|
| **gated** (pplx-decider, [#7](/docs/tutorials/serve-decision-model-vllm-systemone/)) | chance; `label_mass` 0.0003 (broken readout) | **120/120** — but HTTP 501 at the endpoint |
| **invited** (d1-3B, [#9](/docs/tutorials/liquid-d1-decision-model-vllm/)) | 23/120 (19.2%); honest 0.146 confidence | **119/120 (99.2%) — native endpoint, `label_mass` 0.9999** |

Every configuration has now been measured on the same frozen split:

- **Stock + gated**: the baseline nobody should ship.
- **Trained + gated**: accuracy perfection you cannot serve through the decision API (#7's 501).
- **Stock + invited**: served beautifully, wrong about your domain (#9's 23/120).
- **Trained + invited**: served beautifully, right about your domain. *(this tutorial)*

The endpoint whitelist decides whether you *can* serve through the decision API. Fine-tuning decides whether you *should*. Anyone who ships only one of these axes shipped a half-solution.

## Why merge instead of serving the adapter directly

Three options existed for getting a LoRA onto the decision endpoint:

1. **`--lora-modules` at serve time.** vLLM supports adapter serving generally — but the `/v1/systemone` strategies table gates on the *base architecture*, and adapters ride whatever base they attach to. It would probably work; it adds a moving part between "what we trained" and "what answers queries," and this series exists because invisible layers between those two states produce confident nonsense.
2. **`PeftModel.merge_and_unload()` on the full VL class.** The natural call — but our training ran on `Lfm2VlForConditionalGeneration`'s *text tower* (`Lfm2ForCausalLM(text_config)`), because the decision endpoint exercises text-only paths and 12 GB does not fit two towers comfortably. Peft's merge machinery assumes the adapter was attached to the exact module tree it is merging into.
3. **Hand-rolled delta math in the verified key namespace.** `W' = W + (B·A)·(α/r)`, applied tensor-by-tensor, everything else copied through byte-for-byte. Twelve lines of real logic, plus ~40 lines of assertion gates that make a silent key mismatch impossible.

We chose 3, and the gates earned their keep immediately — see "The merge war story" below.

## Step 1: train a rank-16 LoRA on the whitelisted text tower

Full source: [train_d1_lora.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/10-fine-tune-d1-decision-model-vllm/train_d1_lora.py).

Load the checkpoint's safetensors directly, build `Lfm2VlForConditionalGeneration(config)` for the config/tokenizer but take its **text tower** as the training surface via `Lfm2VlForCausalLM(text_config)`, and apply the verified key map (the checkpoint names the text tower `model.language_model.X`; the native text class names it `model.X`). Two handbook rules are hard-asserted, not printed:

```python
assert len(loaded) >= 260, (
    f"only {len(loaded)}/{len(text_sd)} text-tower tensors loaded -- "
    f"key map wrong, model would train from random init. "
    f"skipped sample: {skipped[:5]}")
```

The first smoke run of this harness loaded **0/266** tensors and "trained" a randomly initialized model at loss 12.19 ≈ ln(vocab). An assert caught what a print statement would have celebrated. The telltale numbers thereafter became part of the contract: step-1 loss on a correctly loaded pretrained model opened at **2.52**; a fresh-init model opens at ~11.7. If your curve doesn't start where physics says it must, you are measuring fiction.

Training data uses the frozen generator from #6 (same seed discipline): 800 train pairs, 120-example test split never touched by gradients. Each example renders through vLLM's own rendering path so the prompt the trainer optimizes is byte-identical to what `/v1/systemone` will write at serve time:

```
<|im_start|>user
Route this on-call incident to the owning team.
[Quantly] gimbal health checks are flapping between healthy and down in eu-fra Paged at 04:12 UTC.

Options:
A squad K
B squad O
...
H squad M

Reply with the option code only.<|im_end|>
<|im_start|>assistant
```

with exactly one supervised token — the option letter. Loss masks everything else. Config: LoRA r=16 α=32, q/k/v/o + MLP projections + conv in/out projections → **166 modules, 24.5M trainable params (0.9%)**, bf16, 3 epochs × 800 pairs at LR 1e-4 (grad-accum 8, 300 optimizer steps). GPU steady-state: 6.8 GB at 98% utilization. Full run, wall-clock on the 3080 Ti: roughly twenty minutes. Final loss trace: **2.21 → 0.005** (head-20 vs tail-20 means over 300 steps).

## Step 2: merge — the war story in four gates

This is the section where we show our work rather than the highlight reel. The merge script processes the checkpoint's 707 safetensors keys, swaps in deltas for the 166 LoRA-targeted text-tower linears, and copies everything else (vision tower, embeddings — untouchable by the adapter — plus all 14 aux files including `config.json`, tokenizer artifacts, and Liquid's `prompt.py`) straight through. Four separate assertion failures fired before the first successful merge:

1. **`merged 0 != adapter paths 166`** — lookup returned nothing. We had stripped the checkpoint prefix to bare tails while peft writes adapter paths in the *training model's* namespace (`model.layers.N.conv.in_proj`). Off by a namespace convention → zero hits.
2. **`adapter path collision during namespace bridge`** — our fix double-transformed. Running the key transform over already-correct peft paths, which don't carry the prefix we sliced off unconditionally, collapsed distinct module paths onto shared tails. The collision assert (`len(mapped) == len(pairs)`) is there precisely so this class of bug fails loudly instead of quietly merging a partial delta set.
3. **`unexpected ck namespace: model.layers.0.conv.in_proj`** — asserted the prefix instead of assuming it, and discovered the transform was pointed at the wrong side of the map. Only checkpoint keys may be transformed; peft paths are used as written.
4. **Dump-before-guessing** — finally, we stopped reasoning about namespaces from memory and printed the actual layouts: peft wrote `base_model.model.model.<module>.lora_{A,B}.weight` (332 tensors = 166×2, zero duplicates — peft was blameless throughout); the checkpoint holds `model.language_model.layers.N.<mod>.weight` with **no second `model.`**. The correct bridge: strip `model.language_model.`, drop `.weight`, re-add `model.`. Applied to checkpoint keys only.

The green run reads like this:

```
[stage] adapter alpha=32 r=16 scale=2.0
[stage] adapter module paths: 166
[stage] merged 166 tensors (expect == adapter paths 166)
[stage] wrote models/d1-3b-opsroute: 707 tensors + 14 aux files
[stage] arch: ['Lfm2VlForConditionalGeneration'] | auto_map: True
```

Full source: [merge_d1_lora.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/10-fine-tune-d1-decision-model-vllm/merge_d1_lora.py).

Two lessons generalize beyond this model family:

- **Never transform both sides of a key map.** Pick one canonical namespace and convert one direction only. Every collision/collapse bug above traces to transforming twice or slicing without checking the prefix.
- **Dump, don't remember.** Two of the four failed runs were us guessing at key layouts we could have printed in five seconds. Namespace conventions between checkpoint files, HF class internals, and peft serialization are *not* consistent enough to memorize.

## Step 3: serve it through the front door

The merged directory passes the whitelist unchanged — architecture still `Lfm2VlForConditionalGeneration`, `auto_map` intact — so it boots with no flags beyond the series standard:

```bash
vllm serve ./models/d1-3b-opsroute \
  --served-model-name d1-opsroute \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --logprobs-mode raw_logprobs \
  --trust-remote-code
```

Footprint identical to #9: 10.4 GB reserved at 0.85 utilization on the 12 GB card, startup in under a minute, native `/v1/systemone` accepted with zero registration work. Remember the #9 pre-flight probe: query `available_models` on the strategies table before downloading anything — for a fine-tune of an already-whitelisted model, this is just confirming your merge didn't mutate the architecture declaration.

## Step 4: the frozen test set, end to end

Full source: [eval_d1_finetuned.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/10-fine-tune-d1-decision-model-vllm/eval_d1_finetuned.py).

Same script shape as #9's eval — same 120 examples, same eight-squad option list (so the 12.5% floor applies), same client-side `label_mass` recomputation from raw logprobs (we verify vendor diagnostics, never trust them):

```
D1-3B FINE-TUNED ACCURACY: 119/120 = 0.9917
label_mass median=0.9999 min=0.9995 max=1.0000
confidence median=0.9986 mean=0.9859
```

Against stock (#9, same box, same split): 23/120 at confidence 0.146. Against our #6 router through the *legacy* path: 120/120 — accuracy parity within one example, but the legacy path is where pplx-decider got a 501 and where there are no typed questions, no diagnostics, no contract.

Note what did *not* change when we added the LoRA: the diagnostics stayed honest. `label_mass` moved from 0.99 (confidently right about support tickets) to 0.9999 (correctly right about ops routing) — the adapter sharpened predictions without inflating the readout. Contrast #7's broken model, which reported confident-looking tokens while `label_mass` sat at 0.0003. A well-formed decision head produces diagnostics that track truth; that property survived fine-tuning intact, which is exactly why we can trust 0.9999 here.

### The one miss

Out of curiosity, the single error is worth naming rather than rounding away: one example lands at confidence 0.06 — uniform-ish uncertainty across the eight squads — and picks wrong. It is the *same kind of example* stock d1 got wrong: ambiguous phrasing that genuinely straddles two squads. The fine-tune converted 96 confident-correct outcomes and left behind one honestly-uncertain coin flip. A decision model whose rare failure mode is saying "I don't know" in the only way its diagnostics allow is behaving correctly. Route-to-human on `label_mass < 0.6` catches it for free.

## What this unlocks on consumer hardware

The full-reproduction cost of this tutorial on one card, in order: download 5.9 GB of weights, 20 minutes of training at 6.8 GB VRAM, a deterministic merge pass, boot in under a minute, evaluate. No multi-GPU node, no vendor fine-tuning API, no export step that changes numerics between trainer and server — the served artifact is bit-exact merged weights you can audit tensor-by-tensor against the adapter math.

For anyone evaluating the new wave of small decision/typed-answer models (this family, and the ones cloning its pattern), the evaluation protocol falls out of this arc directly:

1. Probe the server's strategies/whitelist table *before* committing to the model (and check the model's own declared architecture against the registry — #9's warning about architectures that exist in marketing but not in the whitelist).
2. Serve stock, run a frozen domain split, record `label_mass` alongside accuracy — accuracy tells you if it's right, diagnostics tell you if it knows.
3. LoRA the whitelisted text tower, train through the server's own render path so trainer and endpoint agree byte-for-byte.
4. Merge with assertion gates on key coverage in *both* directions (every adapter found a base; every expected module produced a delta).
5. Re-run the same frozen split through the native endpoint — never a side-channel replica of the prompt format.

## Verdict

Being invited and being trained are independent properties, and now we have four measurements proving it. Most teams discover one axis, celebrate, and ship a half-table. The complete cell — **119/120 through the native decision endpoint with 0.9999 label_mass on a 12 GB consumer card** — required both, and each half came from a different tutorial's failure: the 501 taught us to respect the gate, the 23/120 taught us not to trust the invitation.

The scripts below reproduce every number in this post against the same frozen seed. Start with the probe, keep the asserts, dump before you guess.

## Further reading

- [Serve Decision Models on vLLM's /v1/systemone](/docs/tutorials/serve-decision-model-vllm-systemone/) — the 501 origin story and the raw-logprob verification layer
- [The Decision Model That Got Invited](/docs/tutorials/liquid-d1-decision-model-vllm/) — stock d1-3B: the 23/120 foil
- [Fine-Tune a Decision Router with LoRA](/docs/tutorials/fine-tune-lora-decision-router/) — the frozen dataset, the 120/120 trained-but-gated router
- [The Silent Failure Handbook](/docs/tutorials/silent-failure-handbook/) — the bug classes each gate in this post is auditing for
