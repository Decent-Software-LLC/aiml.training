---
title: "Invited at First Sight: A Third-Party Decision Model That vLLM Serves Natively — and Why It Still Scores 18/120"
slug: "amos-decision-model-vllm-native"
prev: docs/tutorials/hf-decision-pipeline-vs-vllm-systemone
sidebar:
  open: true
description: "The Amos 1 family (Apache-2.0, built on Qwen3.5) is the first third-party decision model whose architecture sits inside vLLM's READ_STRATEGIES guest list by construction — no patching, no merge gymnastics. We served amos-1-small natively on vLLM 0.34's /v1/systemone, verified the readout with a garbage-options control, and ran the frozen OPS-ROUTE benchmark through four paths: vLLM, the authors' own server, and their hosted medium and large. All four land near the 12.5% chance floor. The invitation is real; the domain knowledge still isn't."
date: 2026-11-19
lastmod: 2026-11-26
draft: false
weight: 12
type: docs
images: ["og/amos-decision-model-vllm-native.png"]
toc: true
categories: ["Tutorials"]
tags: ["vllm", "transformers", "decision-models", "hf", "12gb", "model-verification"]
---

[Tutorial #7](../serve-decision-model-vllm-systemone/) mapped vLLM's decision guest list (`READ_STRATEGIES`) and showed what it costs to get *onto* it: patching the registry, or merging an adapter into a whitelisted architecture ([#10](../fine-tune-d1-decision-model-vllm/)). [Tutorial #11](../hf-decision-pipeline-vs-vllm-systemone/) showed the two stacks' whitelists are mirror images. This one is about what happens when a model walks up that already fits the list — no patch, no merge, no warning.

The model: **Amos 1** ([`mukaddamzaid/amos-1-*`](https://huggingface.co/collections/mukaddamzaid/amos-1)), a four-size family of decision models released in October under **Apache-2.0**, built on Qwen3.5 (small = 2B, medium = 9B, large = 27B). Their model card claims 76.5% zero-shot on the [typed-decisions benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) — level with Microsoft's closed Microsoft-Decision-1 (79.5%) and above Perplexity's decider (73.9%). If that's true, this is the strongest open-weights decision model line we've seen. And because the family is built on `Qwen3_5ForConditionalGeneration` — an architecture that is *in* vLLM's `READ_STRATEGIES` — vLLM should invite it at first sight.

It does. It scores 18/120 on our frozen benchmark. Both facts are real, and the distance between them is the whole tutorial.

## The setup

- **Model:** `mukaddamzaid/amos-1-small` (2.2B params, 4.12 GiB bf16, Apache-2.0). Also probed: their hosted `amos-1-medium` and `amos-1-large` APIs.
- **Benchmark:** the frozen OPS-ROUTE test split — 120 on-call incidents, eight invented squad names, 12.5% chance floor. Same schema we've used since [#6](../fine-tune-decision-router/): `state` + one `choice` question, criteria = the full team dictionary.
- **Stacks, one at a time on the same 12 GB RTX 3080 Ti:** vLLM 0.34.0 (stock, unpatched) and the authors' own `amos.serve` package (their pinned env: torch 2.14.0, transformers 5.17.0, flash-linear-attention 0.5.2, Python 3.12).
- **Ground truth first:** before promising anything, we checked `model/config.json` — `architectures: ["Qwen3_5ForConditionalGeneration"]` — and grepped `READ_STRATEGIES` in the pinned vLLM ([`entrypoints/openai/decisions/strategies.py`](https://github.com/vllm-project/vllm)): `Qwen3_5ForConditionalGeneration -> NextTokenStrategy`. On the list. (This is the check #9 taught us: read the installed registry, not the docs.)

## Finding 1: vLLM serves it natively — first try, zero patches

```bash
vllm serve models/amos-1-small/model --served-model-name amos --port 8127 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --logprobs-mode raw_logprobs
```

617/617 tensors loaded, engine up in ~40 s, 8.7 GiB resident. A `POST /v1/systemone` with the standard invoice example answers correctly and confidently:

```json
{"answers": {
  "refund": {"type": "noul", "noul": 0.967},
  "team":   {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.828, "technical": 0.053, "sales": 0.119}}}}
```

No registry patch (we needed one in #7), no merge (the #10 route), no 501. The first third-party decision model in our series that vLLM invites outright.

One structural note: the checkpoint ships **no `lm_head` tensor** (`tie_word_embeddings: true`) and its extra decision heads live in a separate `heads.pt` (span/region heads for quote-and-point tasks, not the choice readout). vLLM doesn't care — `NextTokenStrategy` reads option probabilities from the tied output embedding. The authors' own server reads the same signal through a letter-codec (only 52 letter logits computed). Two implementations, one model.

## Finding 2: the readout is real — a garbage-options control

A low benchmark score is only interesting if the plumbing works, so before anything else: a control the model has no reason to fail.

```json
{"state": "The cat sat on the mat.",
 "questions": {"q": {"type": "choice",
   "instructions": "What animal is in the sentence?",
   "criteria": {"banana": null, "giraffe": null, "cat": null}}}}
```

| Server | choice | P(cat) |
|---|---|---|
| vLLM 0.34 `/v1/systemone` | cat | 0.985 |
| authors' `amos.serve` | cat | 0.970 |

The readout is semantic, not structural. Whatever the benchmark says next, it is not a serving bug.

## Finding 3: HF rejects it — the mirror of #11, flipped

In #11, HF *accepted* pplx-decider (with a randomly initialized head — the silent failure) while vLLM 501'd it. With Amos the reflection flips: vLLM invites, HF refuses. `AutoModelForDecision.from_pretrained` on the same directory:

```
/home/agent/aiml-validation/models/amos-1-small/model has no JevForDecision head next to its weights.
```

HF's decision mapping resolves the config to `JevForDecision` but then demands the decision head *next to the weights*; Amos keeps its heads in `heads.pt` and its choice readout in the tied embedding, so the loader walks away. Neither stack is wrong; the checkpoint format just satisfies vLLM's contract and not HF's. If you're picking a decision checkpoint for the HF pipeline, this format is a no-go today; for stock vLLM, it's a first-class citizen.

## Finding 4: the frozen benchmark — four paths, one floor

The same 120 examples, same request schema, four ways to reach an Amos model:

| Path | Accuracy | Confidence (median) |
|---|---|---|
| amos-1-small on vLLM 0.34 (this box) | **18/120 = 15.0%** | 0.083 |
| amos-1-small on `amos.serve` (this box) | 19/120 = 15.8% | 0.095 |
| hosted `amos-1-medium` (9B) | 18/120 = 15.0% | 0.048 |
| hosted `amos-1-large` (27B) | 20/120 = 16.7% | 0.037 |

The chance floor is 12.5%. Scaling the model 2B → 9B → 27B buys ~2 points. It gets worse: the prediction distribution is skewed — `kestrel` predicted 56 times (gold: 20), while `marlin` (gold: 26, the most common team) was predicted **once**. A dumb always-`marlin` baseline would score 26/120. The model isn't just wrong; on this vocabulary it's wrong with a bias.

Two diagnostics make sense of it. First, the confidence numbers are honest: 0.04–0.10 median on an 8-way choice is a model telling you it has no idea — same signature as d1-3B's 0.146 in [#9](../liquid-d1-decision-model-vllm/). Second, the squad names are invented (`kestrel`, `onyx`, `basalt`…) and the service names are invented too (`lantern`, `mothlight`). Nothing in any public training corpus ties `lantern` incidents to `squad K`. The vendor's 76.5% is on four *real-world* workflows (customer service, invoices, security incidents, agent traces); our set is deliberately out-of-distribution by vocabulary — the same information ceiling we documented for d1-3B (23/120 zero-shot) and beat with 300 steps of LoRA in #10.

We did **not** run their typed-decisions benchmark ourselves, so we can't verify the 76.5% claim — we can only say their own server, their hosted API, and stock vLLM all agree with each other and with the chance floor on *our* set. The claim and our number are both consistent with a model that is genuinely strong on general typed decisions and genuinely blind to a private vocabulary.

## Finding 5: diagnostics are server-dependent — don't compare across stacks

Same weights, same request, different diagnostics:

| Signal | vLLM `NextTokenStrategy` | `amos.serve` |
|---|---|---|
| `label_mass` (invoice example) | 0.11–0.62 range across questions | not exposed |
| `confidence` (invoice team) | 0.43 | 0.94 |
| OPS-ROUTE confidence median | 0.083 | 0.095 |

vLLM's `label_mass` sums full-vocabulary probability on the label tokens; for Amos's letter-codec readout roughly half the mass sits elsewhere, so it hovers near 0.5 — *not* the 0.99 d1-3B showed in #9. That's a strategy artifact, not model health. And the two servers compute `confidence` differently (raw vs normalized-within-options): 0.43 vs 0.94 on the same answer. **Rule: diagnostics are only comparable within one server.** Cross-stack, compare argmax and frozen-benchmark accuracy only.

## Footprint (12 GB box)

| Server | Resident | Init |
|---|---|---|
| vLLM 0.34, 4096 ctx, 0.85 util | 8.7 GiB | ~40 s |
| `amos.serve` (their pins, py3.12 venv) | 4.8 GiB | ~30 s |

The 2B model fits either way with room to spare; the 9B medium would need the 24 GB class on this stack mix.

> **Nov 26, 2026 note on `--gpu-memory-utilization 0.85`:** that's our measurement condition, and it's now a *deviation* from stock 0.34 behavior — the release flips the default to 1.0 with extensible KV cache (pool sized from free memory after warmup) and officially recommends dropping the flag ([release notes](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)). For a one-token-per-question decision workload the pool size barely matters; keep an explicit value only when sharing the card or reproducing our 8.7 GiB resident figure. Related 0.34 traps for copy-pasters: `--block-size` default 16 → 64, and `huggingface_hub >= 2.0.0` is now mandatory.

## What this sets up

Amos-1-small is a Qwen3.5 architecture — which means the entire [#6](../fine-tune-decision-router/) LoRA recipe applies to a *natively invited* model with none of #10's merge-namespace gymnastics: train a LoRA on OPS-ROUTE, and the adapter's base is already in `READ_STRATEGIES`. The trained-and-invited cell, but for a checkpoint that never needed inviting. That's the next installment.

Until then, the two-axis map from #9 still holds, one row richer: **invited** is a property of the architecture, **competent** is a property of your data, and a free pass through the whitelist buys you the first and none of the second.

**Download the full source:**
- [eval_amos_vllm.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/12-amos-decision-model-vllm-native/eval_amos_vllm.py) — frozen OPS-ROUTE eval against vLLM `/v1/systemone`
- [eval_amos_native.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/12-amos-decision-model-vllm-native/eval_amos_native.py) — same eval against the authors' `amos.serve`
- [probe_control.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/12-amos-decision-model-vllm-native/probe_control.py) — garbage-options control + prediction-distribution probe
