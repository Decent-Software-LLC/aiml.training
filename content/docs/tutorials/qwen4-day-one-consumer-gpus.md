---
title: "Qwen 4 on Consumer GPUs: Day-One Numbers Before the Benchmarks Arrive"
slug: "qwen4-day-one-consumer-gpus"
prev: docs/tutorials/fine-tune-d1-decision-model-vllm
sidebar:
  open: true
description: "PENDING — skeleton only. Fill every bracket from a real logged run; delete this line and any unfilled bracket before publishing."
date: FILL-weights-day
lastmod: FILL
draft: true
weight: 11
type: docs
images: ["og/qwen4-day-one-consumer-gpus.png"]
toc: true
---

<!-- SKELETON RULES (delete before publish):
- Every number comes from qwen4_dayone_measure.py output (PASTE-ME block) or a
  screenshot we link. No estimates, no vendor numbers presented as ours.
- If vLLM support was missing day-one, THAT is the article's spine (support
  matrix + workaround), same honesty pattern as #7/#9.
- Owner-side hardware (5090/3090) measurements: label as owner-run, keep the
  script + raw block in the repo downloads so it's auditable.
-->

## What shipped (and what didn't)

FILL: exact HF repo IDs seen by the watcher, timestamps (weights-live time vs
this article's run time), license line from the model card, tier table
(Max/Plus/Flash API-only vs 27B open weights — state plainly which are
downloadable).

## Day-one support matrix (the part nobody waited for)

FILL: pre-flight output — architectures + model_type vs vLLM FILL version
registry. If supported: one paragraph, move on. If NOT: this becomes the lead
section (arch names, what's missing, the llama.cpp/SGLang status if checked).

## The runs

### RTX 5090 — Qwen4-27B-FP8

FILL: load wall time, VRAM after load, decode tok/s, config used
(gpu_memory_utilization, max_model_len, quant). Paste the PASTE-ME JSON block
verbatim in a details/collapse.

### RTX 3090 + PolyStrata — MoE tier (optional second act)

FILL: only if owner runs it; Flash-Next-style arch notes (n-gram memory table,
active params) + measured tok/s.

## What the 3080 Ti can't do (and that's the point)

FILL: 27B FP8 weight size vs 12 GB — one honest paragraph feeding the VRAM
calculator row; link /tools/vram-calculator/.

## VRAM calculator update

FILL: add measured row (same pattern as existing measured rows).

## Verdict

FILL: three sentences max. What we'd run in production today, what we'd wait
for (vLLM maturation, non-FP8 weights, benchmarks from others).
