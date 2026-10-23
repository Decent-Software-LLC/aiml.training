---
title: "Fine-Tune Kev-4B: A Calibrated Decision Model on a 12 GB GPU"
slug: "fine-tune-kev-4b-decision-model"
prev: docs/tutorials/evaluate-fine-tuned-llm
sidebar:
  open: true
description: "Take System One's open Kev-4B decision model, fine-tune it on 834 support tickets in 8 minutes on one RTX 3080 Ti, and prove it got better with a paired benchmark: 81.8% vs 66.7% on a held-out set, with the exact VRAM and per-class numbers from our run."
date: 2026-10-20
lastmod: 2026-10-20
draft: false
weight: 4
type: docs
images: ["og/fine-tune-kev-4b-decision-model.png"]
toc: true
author: "K J"
categories: ["Tutorials"]
tags: ["kev", "decision-models", "fine-tuning", "qwen3.5-4b", "system-one", "12gb", "banking77"]
---

> **Download the full source:** [launch-kev-ft.sh](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/04-fine-tune-kev-4b/launch-kev-ft.sh) · [kev-train.sample.jsonl](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/04-fine-tune-kev-4b/kev-train.sample.jsonl) · [README.md](https://github.com/Decent-Software-LLC/aiml.training/blob/main/static/code/tutorials/04-fine-tune-kev-4b/README.md) — the launch script, a sample Kev JSONL record, and setup notes for this tutorial.

## What you'll build

A complete, reproducible pipeline for fine-tuning **Kev-4B** — System One's open decision model — on a domain dataset, then evaluating the tuned model against the stock model on the same held-out tickets. You'll get:

- A working `kev` setup that trains a LoRA + pointer-head decision model on 834 tickets in **~8 minutes** on one 12 GB GPU (RTX 3080 Ti)
- VRAM and step numbers from our actual run, so you know what to expect before you run it
- A **paired benchmark** — the tuned model vs. stock Kev-4B on the same 66 held-out tickets — that shows a clean win: **54/66 (81.8%) vs 44/66 (66.7%)**
- The honest details: one class where the tuned model got worse, and why that's still a good result

The previous tutorials took a general-purpose 4B model and made it behave like your app. This one is different: you start from a model that is already a decision engine — it eats structured context and a set of options, and emits calibrated probabilities over those options — and you teach it your specific decision. That's the shape most production agents actually need, and Kev is one of the few open weights that ships it.

## What is Kev-4B, and why is it worth fine-tuning

Kev is System One's open decision model, built on `Qwen3.5-4B`. Unlike a chat model, its input is not free-form conversation: it's a **decision prompt** — the current state of the world, plus one or more *questions*, each with instructions and a set of **options** (each option has a key and human-readable criteria). Its output is not text you parse; it's a **distribution of probabilities over those options**, produced by a dedicated 256-dimensional pointer head trained for that.

Two things make this useful:

1. **The probabilities are real.** Kev is trained and served with a ranked-probability objective, so the numbers you get back are calibrated, not vibes. You can threshold them, log them for audit, and let the application decide what to do with each outcome — accept, route, escalate.
2. **Fine-tuning it is cheap.** The open weights are a base Qwen3.5-4B plus LoRA adapters plus the pointer head — 496 tensors total. You can warm-start from the stock `jaredpalmer/kev-4b` checkpoint and adapt it to a new decision with a few hundred examples, on a single 12 GB GPU.

That last point is the whole point of this tutorial: **you don't need to train a decision model from scratch, and you don't need a prompt-engineering fight against a chat model's refusal to emit clean probabilities.** You take a model that already knows how to make calibrated choices and you teach it your choices.

## The task

To keep the numbers honest and reproducible, we fine-tune Kev-4B on **banking77** — 11 support-ticket categories from a bank helpdesk — reframed as a single Kev decision:

> Given a customer's ticket text (the *state*), which of the 11 support teams should handle it?

Each ticket becomes one Kev record: the ticket text as `state`, one `choice` question, and 11 options whose `criteria` strings describe each team (e.g. `fraud: "unauthorized transactions, unknown charges, misuse"`). The label is the correct team's key.

We split the data **stratified by class**: 834 training tickets and a held-out set of **66 tickets — exactly 6 per class**. The holdout is never seen during training, and both the tuned and stock models are scored on this exact same set, one ticket at a time. That pairing is what makes the final comparison meaningful (more on that below).

## Setup

Everything runs from a single machine with one 12 GB GPU. Kev's dependencies (PyTorch, Transformers) are the same stack as the LoRA tutorial, so if you've done that one, this is familiar:

```bash
# RTX 3080 Ti (12 GB), Linux, CUDA
nvidia-smi   # confirm: 12 GB, driver >= 595.x

# Kev's environment — pin the CUDA build, same as before
uv venv --python 3.11 /home/agent/aiml-validation/kev/venv
source /home/agent/aiml-validation/kev/venv/bin/activate
pip install "torch==2.13.*" "transformers>=5.19" peft
```

Kev itself is available as a Python package; on our box it lives at `/home/agent/aiml-validation/kev/` and is driven as a module (`python -m kev.train`, `python -m kev.benchmark`). The stock weights (`jaredpalmer/kev-4b`) download from Hugging Face on first use — no API key needed.

## Step 1 — Generate the dataset

The training file is Kev's native JSONL: one record per ticket, each with a `state` and a `questions` dict. A single record looks like this:

```json
{
  "state": "i do not know what this charge is for",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which support team should handle this ticket?",
      "criteria": {
        "card":        "Card: card won't work, won't activate, or delivery issues",
        "fraud":       "Fraud: unauthorized transactions, unknown charges, misuse",
        "topup":       "Top-up & deposits: top-ups and deposits that are missing, slow, or failed",
        "refund":      "Refunds: refunds that are pending, delayed, or missing",
        "transfer":    "Transfers: transfers that are declined, blocked, or slow",
        "fees":        "Fees & charges: unexpected fees, duplicates, or exchange-rate charges",
        "lost_stolen": "Lost or stolen: lost, stolen, or ATM-retained cards",
        "withdraw":    "Withdrawals: cash withdrawals that are declined or partial",
        "account":     "Account: opening, closing, or account security and access",
        "identity":    "Identity: verification, proof of identity, and security checks",
        "fx":          "Currency & exchange: currency conversion and exchange rates"
      },
      "label": "fraud"
    }
  }
}
```

The `criteria` strings are the prompt the model reads to decide between options — they're worth writing carefully, because they *are* your feature engineering. The `label` is the key of the correct option.

The split is deliberately boring: stratified, 6 tickets per class in the holdout, 11 classes. No leakage, no cleverness. If your real task has 300 classes, the same split logic scales; if it has 4, use the whole test set.

Our generation script produced:

- `kev-train.jsonl` — 834 records
- `kev-heldout.jsonl` — 66 records (6 per class)

## Step 2 — Train

The exact training command we ran:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m kev.train \
  --data /home/agent/aiml-validation/kev/kev-train.jsonl \
  --base Qwen/Qwen3.5-4B-Base \
  --init_from jaredpalmer/kev-4b \
  --lr 2e-5 --epochs 2 --batch 1 --accum 8 \
  --dtype bf16 --weights_dtype bf16 \
  --checkpointing 1 --device cuda \
  --out /home/agent/aiml-validation/kev-fin
```

What's happening under the hood, and why each flag matters on a 12 GB card:

- **`--init_from jaredpalmer/kev-4b`** — "delta" mode. Warm-starts the LoRA adapters and the pointer head from the stock checkpoint, resolving **496 tensors** from Hugging Face, pinned by sha256 (`90e817…` for the weights). This is the difference between "train a 4B model from scratch for days" and "adapt a decision head for 8 minutes."
- **`--weights_dtype bf16`** — store the frozen backbone weights in bf16, not fp32. This is the single flag that makes the run fit: fp32 weights on a 4B model alone are ~16 GB, which doesn't fit. bf16 drops that to ~8 GB and leaves room for the 256-dim head, LoRA grads, and activations.
- **`--dtype bf16`** — run the forward pass in bf16 (autocast) with fp32 master weights, so gradients stay numerically stable while the big matrix ops stay fast.
- **`--checkpointing 1`** — activation checkpointing: trade recompute for VRAM. On a 12 GB card it's the difference between fitting and OOM.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — tells the CUDA allocator to grow memory in expandable segments instead of fixed chunks, cutting fragmentation. Cheap insurance on a tight card.
- **`--lora 16`, `--lora_targets all` (default)** — adapter rank 16 across all linear layers. Plenty for a single-decision adaptation.
- **`--head_dim 256` (default)** — the pointer head width. It's the layer that maps the model's final hidden state to a logit distribution over your options, and it trains alongside the LoRA adapters.
- **`--lr 2e-5 --epochs 2 --accum 8`** — the full run was **210 optimizer steps**.

We wrapped this in `tmux` so the run survives SSH drops, and started from a clean `--out` directory so we never clobber a previous run:

```bash
rm -rf /home/agent/aiml-validation/kev-fin
tmux new-session -d -s kevft 'bash launch-kev-ft.sh'
tmux attach -t kevft        # Ctrl-b d to detach
```

### What it cost

From our actual run (`training_metrics.json`):

| Metric | Value |
|--------|-------|
| Wall clock | **467 s (~8 min)** |
| Optimizer steps | 210 (2 epochs × 834 / 8 accum ≈ 208, plus bookkeeping) |
| Tokens processed | 343,219 |
| Loss | **1.08 → 0.53** |
| Peak GPU memory | **9.2 GB** of 12 GB |
| Output artifact | LoRA delta + head, **129 MB** |

Two things to notice:

1. **9.2 GB peak on a 12 GB card.** The margin is real but not huge — about 2.8 GB of headroom. If your GPU is smaller, or you want to batch larger, this is where you'll start to hit OOM. The bf16 flag is doing heavy lifting; don't remove it casually.
2. **The artifact is 129 MB.** That's the entire delta from the stock model — LoRA adapters plus the pointer head. You ship 129 MB, not 8 GB, and you can roll back by deleting it. This is the operational payoff of warm-starting a decision model rather than fine-tuning a chat model.

## Step 3 — Benchmark, paired

This is where most fine-tuning tutorials hand-wave, and where Kev's workflow is unusually clean. Instead of "score the tuned model on some test set," we run **both** the tuned model and stock Kev-4B on the **exact same 66 held-out tickets**, and compare them ticket-by-ticket.

```bash
# Tuned model
KEV_DTYPE=bf16 uv run python -m kev.benchmark \
  --run /home/agent/aiml-validation/kev-fin \
  --data /home/agent/aiml-validation/kev/kev-heldout.jsonl \
  --out /home/agent/aiml-validation/kev-fin/eval-ft

# Stock model (its checkpoint ships a fitted inference temperature;
# the benchmark applies it automatically — it only rescales the
# probability values, not which option wins)
KEV_DTYPE=bf16 uv run python -m kev.benchmark \
  --run jaredpalmer/kev-4b \
  --data /home/agent/aiml-validation/kev/kev-heldout.jsonl \
  --out /home/agent/aiml-validation/kev-fin/eval-base
```

`KEV_DTYPE=bf16` is required on a 12 GB card for the same reason as at train time — fp32 inference of a 4B model doesn't fit. Both runs score the same 66 tickets, so the accuracy comparison is apples to apples; the stock checkpoint's fitted temperature only rescales its probability output, not which option is the argmax.

### The result

On the 66-ticket holdout:

| | Correct | Accuracy |
|---|---|---|
| **Tuned (kev-fin)** | **54 / 66** | **81.8%** |
| **Stock (kev-4b)** | **44 / 66** | **66.7%** |

That's a **+15.1 point** improvement. But raw accuracy on 66 samples is too coarse to trust on its own, so we look at the paired flips — which tickets did one model get right and the other wrong:

- **13 tickets** the tuned model got right that stock got wrong
- **3 tickets** stock got right that the tuned model got wrong
- 41 tickets both got right, 9 both got wrong

With 13 wins and 3 losses on a paired test, McNemar's test gives **p ≈ 0.02** — the improvement is statistically significant at the 5% level, not a sampling artifact. That's the "clean result" bar we hold this site's tutorials to.

### Per-class, so you can see where it helps

| Class | n | Tuned | Stock | Tuned won |
|-------|---|-------|-------|-----------|
| card | 6 | 6 | 2 | 4 |
| topup | 6 | 6 | 4 | 2 |
| fraud | 6 | 6 | 5 | 1 |
| fees | 6 | 5 | 5 | 1 |
| fx | 6 | 6 | 5 | 1 |
| identity | 6 | 6 | 5 | 1 |
| lost_stolen | 6 | 3 | 3 | 1 |
| refund | 6 | 5 | 4 | 1 |
| transfer | 6 | 3 | 2 | 1 |
| account | 6 | 2 | 3 | 0 |
| withdraw | 6 | 6 | 6 | 0 |

Two things the per-class table tells you that the headline number hides:

1. **The wins are concentrated in the classes the model was worst at.** `card` went from 2/6 to 6/6; `topup` from 4/6 to 6/6. The fine-tune fixed the specific confusion patterns (card vs. fees vs. topup are easy to mix up from ticket text alone) rather than uniformly nudging everything.
2. **One class got worse: `account`, 3/6 → 2/6.** This is the honest part of the result. With only 6 held-out tickets per class, single-ticket swings are noise, but it's a reminder that a +15 point headline is a *mean*, not a guarantee — a production rollout should still monitor per-class recall, not just overall accuracy.

## Why this shape matters for production agents

If you've built agent systems, you already know that the hard part is rarely "can the model produce an answer" — it's "can it produce an answer *we can act on deterministically*." Kev's design is built for that:

- **Structured input.** State + options + criteria. No prompt-formatting fragility, no "did the model follow the JSON schema this time?" You build the prompt once in code and it's stable.
- **Calibrated output.** The probability distribution over options is the interface. Your application can say "if `fraud` > 0.7, route to the fraud team; if 0.3–0.7, ask the customer one clarifying question; below 0.3, treat as noise." That logic lives in *your* code, versioned and tested, not in a prompt.
- **Cheap to adapt.** 129 MB of delta, 8 minutes of training, 9.2 GB of peak VRAM. You can retrain per-domain (support, triage, moderation, routing) without a new training rig, and roll back by deleting the adapter.
- **Paired evaluation is the right question.** The comparison in this tutorial — same holdout tickets, two models, compared one at a time — is what tells you whether a change *helped or hurt specific inputs*, not just whether the aggregate score moved. Kev's own tooling bakes this in: `kev.benchmark`'s `paired_flip` field scores contrastive pairs (sibling tickets that differ in state, testing whether the model actually reads the state rather than pattern-matching a prior), and `kev.compare` runs a record-clustered paired bootstrap across two runs. Either way, the question is the same: which inputs changed, and for the better or worse.

## Where to go from here

This tutorial covered the full loop: dataset → train → paired eval. The next steps that make this production-shaped:

- **Serve the tuned model.** `kev.benchmark` can hit a remote run (`--remote`), which means the same 129 MB adapter can be served as an API and evaluated identically from a client machine. See the vLLM serving tutorial for the serving-side setup.
- **Expand the holdout.** 66 tickets is enough to show a direction; a production rollout needs hundreds per class. The stratified split logic in this tutorial scales directly.
- **Add a second question.** Kev records can carry multiple questions per state — e.g. "which team" *and* "how urgent" in one call. That's how you turn a single classifier into a small decision system without adding latency.
- **Track the probability distribution, not just the argmax.** The calibrated probabilities are the interesting part. Log them, plot them over time, and alert on drift — that's a free monitoring layer you don't get from a chat model.

## The bottom line

Kev-4B is a rare case of an open weights model that ships the exact shape most production agents need: a calibrated, structured-input decision engine that's cheap to adapt. On a single 12 GB GPU, we took a stock model that scored 66.7% on a held-out set of 66 support tickets, fine-tuned it on 834 more in 8 minutes, and got 81.8% — with the improvement concentrated in the classes it was worst at, and a paired test that says the win is real.

The 129 MB delta, the 9.2 GB peak, and the per-class table above are all from the actual run. If your domain is "given this input, pick one of N options and tell me how confident you are," this is the model to start from.

## Version pins and reproduction

Everything in this tutorial was run against:

- **Base model**: `Qwen/Qwen3.5-4B-Base`
- **Stock checkpoint**: `jaredpalmer/kev-4b` (pinned by sha256 at train time; 496 tensors)
- **PyTorch**: `2.13.*` (CUDA build)
- **Transformers**: `>=5.19`
- **GPU**: NVIDIA RTX 3080 Ti, 12 GB, driver 595.99.02
- **Dataset**: 834 train / 66 holdout, stratified 6 per class across 11 classes (banking77, Kev-reframed)
- **Run config**: 2 epochs, lr 2e-5, LoRA r=16, grad accum 8, `--weights_dtype bf16`
- **Evaluation**: `kev.benchmark` with `KEV_DTYPE=bf16` on both runs; the stock checkpoint's fitted inference temperature applied automatically (probability rescaling only)

The complete training config, run metrics, and per-ticket evaluation rows are on file with us — if any number above doesn't check out, ask and we'll pull it out.
