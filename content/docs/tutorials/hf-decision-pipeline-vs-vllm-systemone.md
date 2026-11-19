---
title: "Two Stacks, One Decision: HF's pipeline('decision') vs vLLM's /v1/systemone, Measured on the Same Models"
slug: "hf-decision-pipeline-vs-vllm-systemone"
prev: docs/tutorials/fine-tune-d1-decision-model-vllm
next: docs/tutorials/amos-decision-model-vllm-native
sidebar:
  open: true
description: "Hugging Face formalized decision models in November with pipeline('decision'), AutoModelForDecision, and its own /v1/systemone server. We ran the same frozen OPS-ROUTE benchmark through both stacks on the same 12 GB box. The support matrices are mirror images — and HF's acceptance of Perplexity's pplx-decider turned out to be a silent failure: a randomly initialized decision head scoring exactly the chance floor."
date: 2026-11-18
lastmod: 2026-11-18
draft: false
weight: 11
type: docs
images: ["og/hf-decision-pipeline-vs-vllm-systemone.png"]
toc: true
categories: ["Tutorials"]
tags: ["vllm", "transformers", "decision-models", "hf", "12gb", "model-verification"]
---

[Tutorial #7](../serve-decision-model-vllm-systemone/) established that vLLM's `/v1/systemone` keeps a guest list: architectures in `READ_STRATEGIES` get native decision serving, everyone else gets a 501. [Tutorial #9](../liquid-d1-decision-model-vllm/) and [#10](../fine-tune-d1-decision-model-vllm/) mapped what happens when you're on the list. Eight days ago, Hugging Face [made decision models a first-class category](https://huggingface.co/blog/decision) — `pipeline("decision")`, `AutoModelForDecision`, a hub tag, and yes, its own `/v1/systemone` endpoint in `transformers serve`.

Same task, same frozen benchmark, same 12 GB RTX 3080 Ti, two stacks. The result is a pair of **mirror-image support matrices** — and one finding we nearly missed: HF *accepts* the model vLLM rejects, and that acceptance is a silent failure.

## The setup

- **Model A:** `perplexity/pplx-decider-flash-v1.2` — the decision model from [#6](../fine-tune-decision-router/) and [#7](../serve-decision-model-vllm-systemone/) that vLLM gates with a 501.
- **Model B:** `LiquidAI/d1-3B` — the model vLLM invites ([#9](../liquid-d1-decision-model-vllm/), [#10](../fine-tune-d1-decision-model-vllm/)).
- **Benchmark:** the frozen OPS-ROUTE test split (120 examples, eight squads, 12.5% chance floor) — identical request schema across stacks: `state` + one `choice` question, criteria = the full team dictionary.
- **Stacks:** vLLM 0.34.0 (fresh venv) and transformers 5.23.0 (fresh venv — the first release line with the decision pipeline). Both on the same box, one at a time.

## Finding 1: HF accepts pplx-decider — and that's the bug

vLLM refuses pplx-decider outright: architecture `Qwen3_5Model` isn't in `READ_STRATEGIES`, request returns 501. Loud, honest, done.

HF's `AutoModelForDecision` does something much more interesting: it **loads it**. The config resolves to `JevForDecision` (Jev being the original decision-model architecture from TypeSafe's System One line). It also prints a warning we almost scrolled past:

```
notes:
- MISSING:  those params were newly initialized because missing from the checkpoint.
```

The missing parameter is `proj.bias` — **the decision head's projection bias**. The checkpoint doesn't carry it, so transformers initializes it randomly and hands you a working-looking pipeline. We ran the frozen test set through it:

| Stack | pplx-decider accuracy | Confidence |
|---|---|---|
| vLLM `/v1/systemone` | refused (501) | — |
| HF `pipeline("decision")` | **15/120 = 12.5% — exactly the chance floor** | median 0.055 |

15/120 on an 8-way choice is not a bad score; it is *the* score of a coin flip. The randomly initialized head erased everything the checkpoint learned. This is [Silent Failure Handbook](../silent-failure-handbook/) bug class #1 — "dead runs that look successful" — in a brand-new pipeline, caught only because we had a frozen benchmark and read the warning line. HF's own load report is the tell: **if `AutoModelForDecision` prints MISSING for anything in the decision head, stop.**

To be fair to HF: the model card for pplx-decider doesn't declare a `JevForDecision`-compatible head, and the mapping is config-based (`Qwen3_5Config` is in `MODEL_FOR_DECISION_MAPPING` because *some* Qwen3.5 checkpoints are decision models). The design assumption — config class implies decision checkpoint — doesn't hold for this one. Our handbook rule generalizes to the new pipeline: **verify the head loaded, not just that the model did.**

## Finding 2: HF's own `/v1/systemone` server 500s on the same model

The blog shows `transformers serve` exposing `/v1/systemone` with the same request shape we've been using since #7. We started it (port 8126, one command) and posted the identical body:

```
ValueError: Could not load any image processor for pplx-decider-flash-v1.2
```

The serving path calls `AutoProcessor` unconditionally ([`serving/utils.py` `_resolve_model`](https://github.com/huggingface/transformers)), which for a `qwen3_5` config demands an image processor the checkpoint doesn't ship. The pipeline works (Finding 1); the server doesn't. Day-one rough edge, filed in our head for the pin note.

## Finding 3: the matrices are mirror images

The same probe in reverse — d1-3B, vLLM's favorite — against HF:

```
AutoModelForDecision ERR: Unrecognized configuration class Lfm2VlConfig
Model type should be one of: Gemma4, Jev, Qwen2, Qwen3_5, Qwen3_5Moe, Qwen3_5Text
```

HF's decision mapping covers Gemma4, Jev, and the Qwen3.5 family — **no Liquid archs at all**. The model vLLM serves natively at 119/120 ([#10](../fine-tune-d1-decision-model-vllm/)) doesn't load as a decision model in HF at all. And our merged 120/120 router from #6-era experiments? HF rejects it too: `Qwen3_5ForCausalLM` is in the mapping family, but the loader demands a decision head next to the weights, and a plain merged LoRA has none.

The complete picture, one table:

| Model | vLLM 0.34 `/v1/systemone` | HF 5.23 `pipeline("decision")` |
|---|---|---|
| pplx-decider-flash | **501** (arch not in `READ_STRATEGIES`) | loads as `JevForDecision`, **random head → 12.5% floor** |
| d1-3B (stock) | invited — 23/120 ([#9](../liquid-d1-decision-model-vllm/)) | **rejected** (`Lfm2VlConfig` not in decision mapping) |
| d1-3B + our LoRA | invited — **119/120** ([#10](../fine-tune-d1-decision-model-vllm/)) | rejected (same config gate) |
| d1-2B | blocked (`D1Model` missing both tables, [#9 update note](../liquid-d1-decision-model-vllm/)) | rejected |
| our merged Qwen3.5 router | 501 (no strategy) | rejected (no decision head next to weights) |

Every cell measured on one box, all scripts below. The pattern: **each stack admits the other stack's flagship and refuses its own gatekeeper's favorite.** There is no "decision model support" in the abstract — there are two whitelists, they overlap on the Jev/Qwen3.5 family, and the category name HF just standardized hides that.

## What this means for choosing a stack

- **Serving Liquid d1 (the current open-weights decision models with real heads):** vLLM only, today. HF has no Liquid support in the decision path.
- **Serving Jev-family / Qwen3.5-based decision checkpoints:** both stacks, but on HF **read the load report** — a MISSING decision-head parameter means you're benchmarking noise.
- **Serving pplx-decider:** neither stack does it correctly today. vLLM's 501 is the more honest failure.
- **Fine-tuned routers (our 120/120 pattern):** vLLM via merge + whitelist ([#10](../fine-tune-d1-decision-model-vllm/)); HF's decision-training path (`AutoModelForDecision` + cross-entropy over options, TRL support [in review](https://github.com/huggingface/trl/pull/8040)) is the obvious next experiment — it trains the head natively instead of bolting one on.

## Pins and artifacts

vLLM 0.34.0 stable (note: the 0.34.0rc4 tags on GitHub were never published to PyPI — stable landed directly; pin watchers keyed on rc tags miss it), transformers 5.23.0, RTX 3080 Ti 12 GB, CUDA 13.0. Eval scripts:

- [eval_hf_pplx.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/11-hf-decision-pipeline-vs-vllm-systemone/eval_hf_pplx.py) — the 12.5%-floor run, warning-line gate included
- [probe_matrix.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/11-hf-decision-pipeline-vs-vllm-systemone/probe_matrix.py) — both registries, one script, run before you download anything

*Next: the HF-native training path — `AutoModelForDecision` + cross-entropy over options on OPS-ROUTE, then the same checkpoint through both stacks. That's the first checkpoint with a real head on both sides of the table.*
