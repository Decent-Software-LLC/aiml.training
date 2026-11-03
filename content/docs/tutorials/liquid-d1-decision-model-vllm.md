---
title: "The Decision Model That Got Invited: Liquid d1 on vLLM's /v1/systemone (and Why It Still Failed Our Benchmark)"
slug: "liquid-d1-decision-model-vllm"
prev: docs/tutorials/silent-failure-handbook
sidebar:
  open: true
description: "vLLM 0.33 whitelists Liquid's d1-3B for native decision serving — choice, yes/no, and score questions answered with honest calibration diagnostics. We served it on a 12 GB card, watched it route a support ticket at 0.98 confidence, then ran it on the frozen OPS-ROUTE test set from tutorial #6: 23/120. Being a real decision model gets you the endpoint. It does not get you your domain."
date: 2026-11-03
lastmod: 2026-11-03
draft: false
weight: 9
type: docs
images: ["og/liquid-d1-decision-model-vllm.png"]
toc: true
author: "K J"
categories: ["Tutorials"]
tags: ["vllm", "serving", "decision-models", "liquid-ai", "12gb", "model-verification"]
---

> **Download the full source:** [d1_systemone_demo.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/09-liquid-d1-decision-model-vllm/d1_systemone_demo.py) · [eval_d1_opsroute.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/09-liquid-d1-decision-model-vllm/eval_d1_opsroute.py) — runnable versions of everything below.

## The short version

[Tutorial #7](../serve-decision-model-vllm-systemone/) ended with our fine-tuned router locked out of vLLM's structured-decision endpoint: `pplx-decider` isn't in the architecture whitelist, and the workaround readout returned chance accuracy with a `label_mass` of 0.0003 telling us so. The obvious follow-up question: what does the endpoint look like when the model *is* invited — a model actually built to decide?

vLLM 0.33 whitelists Liquid AI's **d1-3B** (`Lfm2VlForConditionalGeneration` → `LiquidStrategy`). We served it on the same 12 GB RTX 3080 Ti, asked three typed questions about a support ticket in one call, and got textbook behavior: routing at **0.98** confidence, urgency at **0.80**, severity scored **1.69/3**, all three answered for **three output tokens**, with `label_mass` 0.98–0.99 and `argmax_is_label: true` on every question.

Then we ran it — untrained, zero-shot — on the frozen OPS-ROUTE test set from [tutorial #6](../fine-tune-decision-router/), the same 120 examples our router LoRA aces at 120/120:

| Model | Endpoint status | OPS-ROUTE test accuracy | Diagnostics honesty |
|---|---|---|---|
| pplx-decider (stock) | gated out (501) | 14/120 | label_mass 0.0003 — readout broken |
| pplx-decider + our LoRA | gated out (501) | **120/120** | n/a (custom readout) |
| **d1-3B (stock)** | **invited, first-class** | **23/120** | label_mass 0.99, confidence median **0.146** |

A purpose-built, vendor-benchmarked decision model scored **at chance** on our domain. But unlike the broken readout in #7, d1 *knew it*: median confidence 0.146 across 120 guesses. That contrast — invited-but-untrained vs trained-but-gated, fake certainty vs honest uncertainty — is the whole article. **Fine-tuning is still the moat.**

## What d1 is, and what it costs

Liquid AI ships d1 as a "decision model": instead of generating an answer, it reads the probability of each option's label token in one forward pass. d1-3B is LFM2.5-VL-3B post-trained to answer decisions as its first token. Two facts matter for us:

- **License:** LFM Open License v1.0 — commercial use is free under $10M annual revenue. Fine for most readers; read it if you're not most readers.
- **Footprint on our box:** weights 5.9 GB on disk, ~6.6 GB VRAM at `--gpu-memory-utilization 0.85` with a 4096-token context, leaving **3.3 GiB of KV cache**. Engine init 24 s. This is the "runs where Qwen3.5-4B can't even load" size class — a decision model that fits a laptop GPU is the category's pitch, and the numbers hold.

One pin warning before you copy the demo: **the newer d1-2B (Oct 27) does not work on vLLM 0.33.** Its `config.json` declares architecture `D1Model` / model type `lfm3_vl`, and neither appears in 0.33's model registry or its `READ_STRATEGIES` table. Liquid's own runtime matrix says "coming soon" for vLLM. This is the general rule from the [Silent Failure Handbook](../silent-failure-handbook/): check the checkpoint's declared architecture against the serving stack's whitelist *before* promising a demo. `curl https://huggingface.co/LiquidAI/d1-2B/raw/main/config.json` takes ten seconds and saves the afternoon.

## Serving d1 on 12 GB

The serve command is unremarkable, which is the point — no plugin, no custom head, no fork:

```bash
vllm serve LiquidAI/d1-3B \
  --served-model-name d1 \
  --logprobs-mode raw_logprobs \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

`--logprobs-mode raw_logprobs` is the only decision-specific flag (the endpoint needs raw label probabilities; #7 covers why). Both decision endpoints are live on the server: `/v1/systemone` (vLLM-native) and the newer OpenAI-compatible `/v1/decisions`.

Box gotchas we hit, since we run everything on one 12 GB card: vLLM's Triton JIT needs CPython headers and our container wipes `/tmp` on every recreation — `apt-get download libpython3.11-dev && dpkg -x libpython3.11-dev_*.deb /tmp/pyhdr` then `CPATH=/tmp/pyhdr/usr/include:/tmp/pyhdr/usr/include/python3.11`. (New wrinkle: `dpkg -x` into bare `/tmp` now fails with tar permission errors; extract into a subdir.)

## The demo: three questions, three tokens

The request from `d1_systemone_demo.py` — one state, three typed questions:

```python
req = {
  "model": "d1",
  "state": "Ticket #4412: 'I was charged twice for my subscription this "
            "month and the second charge went to my old card that I cancelled.'",
  "questions": {
    "route":   {"type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {"billing": "charges, refunds, invoices, payment methods",
                             "tech": "bugs, errors, broken features",
                             "account": "logins, profile settings, cancellations",
                             "sales": "upgrades, plans, enterprise inquiries"}},
    "urgent":  {"type": "noul",
                "instructions": "Does this customer need money moved back before their next statement?",
                "criteria": {"true": "a refund or reversal is time-sensitive",
                             "false": "no time pressure on funds"}},
    "severity": {"type": "score",
                 "instructions": "How severe is this issue?",
                 "criteria": ["cosmetic", "minor", "moderate", "severe"]},
  },
}
```

The response (verbatim from our run):

```json
{
  "answers": {
    "route":    {"type": "choice", "choice": "billing",
                 "probabilities": {"billing": 0.9795, "tech": 0.0085,
                                   "account": 0.0062, "sales": 0.0058},
                 "confidence": 0.9729},
    "urgent":   {"type": "noul", "noul": 0.7982,
                 "probabilities": {"yes": 0.7982, "no": 0.2018},
                 "confidence": 0.7817},
    "severity": {"type": "score", "score": 1.691,
                 "legend": {"0": "cosmetic", "1": "minor", "2": "moderate", "3": "severe"},
                 "probabilities": {"0": 0.033, "1": 0.313, "2": 0.584, "3": 0.070},
                 "confidence": 0.5738}
  },
  "usage": {"input_tokens": 242, "output_tokens": 3},
  "diagnostics": {
    "route":    {"label_mass": 0.9932, "argmax_is_label": true},
    "urgent":   {"label_mass": 0.9794, "argmax_is_label": true},
    "severity": {"label_mass": 0.9818, "argmax_is_label": true}
  }
}
```

Read the diagnostics first — that's the habit #7 built. In #7, `label_mass 0.0003` meant the model's probability mass was nowhere near the option tokens: the readout was measuring noise. Here every question carries **98–99% of its mass on the actual labels** and the argmax lands on a label. The instrument is honest. Now you're allowed to read the probabilities.

Notice what the score question does: it doesn't pick a bucket, it returns the *expected value* across the ordered legend (1.691 of 3 — between "minor" and "moderate"). For thresholding ("escalate if severity ≥ 2") that's strictly more useful than a hard pick, and the per-bucket probabilities are still there if you want them.

## The cold shower: 23/120 on our own benchmark

Vendor benchmarks say d1-3B is strong at decisions (Liquid's Decision Bench: 83.6). So we ran it against the one benchmark we own: the **OPS-ROUTE frozen test split** from tutorial #6 — 120 on-call incidents routed by a hidden service×region chart with priority overrides, using team codenames no pretrained model has ever seen. The dataset generator is public; the point of the design was that the routing knowledge is *only* learnable from labelled examples.

`eval_d1_opsroute.py` posts each incident as a `choice` question over the four squads. Result: **23/120 = 19.2%** — at or below chance for a 4-way task.

But look at the calibration, because this is where the article earns its keep:

| Signal | d1-3B on OPS-ROUTE | What it means |
|---|---|---|
| accuracy | 19.2% | no domain knowledge (expected — codenames are unguessable) |
| `label_mass` median | 0.907 | readout is real; probabilities mean what they say |
| confidence median | **0.146** | the model is telling you it doesn't know |

Compare #7's broken pplx-decider readout: label_mass 0.0003 — the *instrument* was lying. Here the instrument works perfectly and reports genuine uncertainty. A decision model you can distrust correctly is a deployable component; one that confabulates certainty is not.

That gives you the two-axis map this series has been building:

- **Trained but gated** (our #6 router): knows the domain, can't use the endpoint.
- **Invited but untrained** (d1 stock): uses the endpoint beautifully, doesn't know the domain — and says so.
- The obvious third cell — **trained and invited** — is the actual engineering project: fine-tune a whitelisted architecture on your routing data. d1-3B is whitelisted *today*, its strategy reads the first token the way the checkpoint's own `prompt.py` does, and Liquid documents fine-tuning as a supported path. We'll run that experiment next; the frozen test set and this eval script make the before/after a one-command comparison.

## What to take from this

1. **Check the whitelist before the weights.** `READ_STRATEGIES` in `vllm/entrypoints/generate/structured_decisions/strategies.py` is the guest list. d1-3B is on it in 0.33; d1-2B isn't yet, regardless of what the release blog says.
2. **Diagnostics first, always.** `label_mass` near 1 and `argmax_is_label: true` before you trust a single probability. They cost zero extra tokens.
3. **A decision model's benchmarks are not your benchmarks.** 83.6 on the vendor's enterprise suite, 19.2 on yours. The gap is domain knowledge, and domain knowledge is exactly what fine-tuning sells.
4. **Honest low confidence is a feature to evaluate for, not a bug.** Median 0.146 on OOD data is what you want a decision model to do; it's the difference between a component and a coin flip.

Pins: vLLM 0.33.0, `LiquidAI/d1-3B` (LFM Open License v1.0), RTX 3080 Ti 12 GB, CUDA 13.0. The 0.32 pin from tutorials #1–#8 still serves everything else; 0.33 is CUDA-13-only wheels, which fit our box unchanged.

*Next in the series: fine-tuning a whitelisted decision architecture on OPS-ROUTE and running the frozen test set through `/v1/systemone` end to end.*
