---
title: "Teach a Decision Model Its Tribal Knowledge: LoRA From Chance to 100 Percent Routing"
slug: "fine-tune-decision-router"
prev: docs/tutorials/end-to-end-lora-pipeline
sidebar:
  open: true
description: "We fine-tuned Perplexity's Apache-2.0 pplx-decider-flash-v1.2 twice on one 12 GB card. On benchmarks the teacher already knows, our LoRA lost (macro-F1 -0.8 pt). On invented routing conventions it went from 11.7% (chance) to 120/120 — and three silent bugs taught us why."
date: 2026-10-25
lastmod: 2026-10-25
draft: false
weight: 6
type: docs
images: ["og/fine-tune-decision-router.png"]
toc: true
author: "K J"
categories: ["Tutorials"]
tags: ["fine-tuning", "lora", "qlora", "decision-models", "qwen3.5-4b", "12gb", "model-verification"]
---

> **Download the full source:** [gen_opsroute_dataset.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/06-fine-tune-decision-router/gen_opsroute_dataset.py) · [train-router-lora.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/06-fine-tune-decision-router/train-router-lora.py) · [eval-router-probe.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/06-fine-tune-decision-router/eval-router-probe.py) — runnable versions of everything below.

## The short version

Fine-tuning gets sold as "make the model better." It doesn't. Fine-tuning **adds knowledge a model cannot obtain any other way** — and where that knowledge is already implicit in pretraining, a LoRA can only add noise. We proved both halves of that sentence on one RTX 3080 Ti, same recipe, same 4-bit QLoRA config, twice:

| Experiment | Stock model | Our LoRA | Verdict |
|---|---|---|---|
| Frozen intent benchmarks (BANKING77 / CLINC150+OOS macro-F1) | 0.9015 / 0.9329 | 0.8935 / 0.9277 | **We lost** half a point on its home turf |
| Invented routing conventions ("OPS-ROUTE") | 11.7% (= chance floor) | **120/120 = 100%** | Perfect, confident, generalizes to unseen brand names |

Same card, same base checkpoint, same adapter rank. The difference is whether the task contains facts that *have to be taught*. Getting there took five training runs because three separate bugs each made a result look real when it wasn't. Those bugs are half the tutorial.

## The setup

Base model: `perplexity-ai/pplx-decider-flash-v1.2` — an Apache-2.0 decision classifier distilled from **Qwen/Qwen3.5-4B**, the same backbone tutorials [#1 through #5](../end-to-end-lora-pipeline/) use. It reads a structured request (state + typed questions) and emits per-option probabilities through a small readout head instead of generating tokens. No download gating, weights SHA-256-pinned via the release manifest.

Hardware: a single RTX 3080 Ti (12 GB). Stack pinned for every stage: torch 2.13.0+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, Triton kernels for the gated-delta-rule layers (with a pure-PyTorch fallback — more on that below).

Training used the QLoRA recipe: NF4 4-bit base (`bnb_4bit_quant_type="nf4"`, double-quant, bf16 compute), LoRA r=16 / α=32 on attention **and** MLP projections **plus** the readout head trained unquantized — **33.1M trainable parameters**, peak CUDA memory 9.6 GiB with gradient checkpointing on. Backbone drops from 9.08 GB to ~3.1 GB; everything fits on the consumer card that [tutorial #1](../lora-fine-tuning-qwen3.5-4b/) started on.

## Act 1 — Measuring what the teacher already knows

Before tuning anything we established a baseline with a **frozen, third-party eval**: the public splits of BANKING77 (3,080 requests) and CLINC150+OOS (5,500) from the Decision Index suite, rebuilt from upstream sources pinned by SHA-256 in the reproduction kit's manifest ([apolinario/decision-index](https://github.com/apolinario/decision-index), MIT). Two rules we imposed on ourselves, both load-bearing:

1. **Eval-only.** Test-split rows never touched training, ever. (The kit's own rule; breaking it turns your score into fiction.)
2. **Kit-native metrics.** Macro-F1 with the suite's published chance constants — not our own invented accuracy flavor.

Baseline (stock weights, 8,580 requests, reference inference path, 294.6 ms/request):

| Benchmark | n | Accuracy | Macro-F1 | Chance | Skill |
|---|---|---|---|---|---|
| BANKING77 | 3,080 | 0.9023 | 0.9015 | 0.0127 | 0.9002 |
| CLINC150+OOS | 5,500 | 0.9145 | 0.9329 | 0.006 | 0.9325 |

That is close to board-topping territory — unsurprising for a model whose card says intent routing was a distillation target. We trained anyway (LoRA r=16, LR 1e-4, 6,000 examples balanced across the two task families — train splits only), just under four hours of GPU. Re-evaluated on the *frozen* test rows: macro-F1 **0.8935** and **0.9277**. We lost half a point on both. Calibration moved the wrong way too (expected calibration error, 15 bins, computed offline from stored probability vectors):

| Benchmark | ECE stock | ECE LoRA | avg top-1 stock | avg top-1 LoRA |
|---|---|---|---|---|
| BANKING77 | 0.1109 | **0.1257** | 0.8451 | 0.8253 |
| CLINC150+OOS | 0.0753 | **0.0906** | 0.8950 | 0.8746 |

Three measurements (accuracy, F1, calibration) agreeing on "no gain, slight harm" is not an unlucky seed; it's information. Our training distribution sat *inside* whatever corpus Perplexity distilled from. When the teacher already embodies the task, additional gradient steps wander off its optimum. The publishable form of this experiment would have been a one-line negative result — except that our own val numbers said otherwise (below), and chasing that contradiction uncovered three bugs that apply to far more than decision models.

## Bug #1 — a trainer that silently took zero optimizer steps

First OOD attempt (convention routing, Act 3) trained for ~18 minutes and finished at chance accuracy. Loss curve looked like noise. We initially blamed the model. The actual bug was four lines of our training loop:

```python
if (step + 1) % args.accum == 0:   # requires step % 8 == 7 ...
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    step += 1                       # ...but `step` only advances INSIDE this branch
```

At `step=0` the gate needs `step ≡ 7 (mod 8)` — and nothing outside the branch ever increments `step`, so the condition is unreachable. Every backward pass computed gradients; **nothing ever applied them**. GPU pinned at 100%, wall time burned, weights frozen forever. Worse, the same pattern silently afflicted the Act-1 run two days earlier: its `train-log.json` records `"log": []` — not a single optimizer step reached the logging gate either. Which means Act 1's "LoRA lost by half a point" was really "an identical-weight model reloaded through the PeftModel wrapper lost half a point," i.e. bf16 tie-breaking wobble flipping near-margin argmaxes. The negative conclusion happened to survive anyway (three metrics, later reproduced) — but we didn't know that yet.

Verification discipline added afterward, and worth stealing verbatim:

```python
# after launch, BEFORE trusting any result:
#  1. grep the log for 'step' lines within the first N minutes
#  2. if absent, stop — you are warming kernels or deadlocked, not learning
pgrep -f train.py && sleep 120 && grep -c '^step' train.log   # must be > 0
```

And use a **separate microbatch counter** for accumulation gating (`mb += 1` per backward; step when `mb % accum == 0`). Never gate a counter on itself.

## Bug #2 — gradient checkpointing that checkpoints nothing

On a 12 GB card, full-precision backward dies while 4-bit + activations should fit. Our peak kept hitting 11+ GiB despite calling:

```python
backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})```

and still watching `torch.cuda.max_memory_allocated()` balloon past 11 GiB during backward. The trap: HuggingFace's `GradientCheckpointingLayer` gates recomputation on **two** flags — `module.gradient_checkpointing` *and* `module.training`. Our Decider wrapper leaves everything in `eval()` mode (correct for inference!), so checkpointing was "enabled" but inert, and every layer kept its activations. Two additions fixed it:

```python
pefted = get_peft_model(backbone, lora_config)   # enable AFTER wrapping:
pefted.gradient_checkpointing_enable(            # enabling while all params are
    gradient_checkpointing_kwargs={"use_reentrant": False})  # frozen can leave it inert
pefted.train()          # THIS is what most HF-based training loops inherit for free
                        # and custom loops forget: layers are ModuleList members you
                        # must flip to training mode yourself
```

Peak dropped from guaranteed-OOM to ~10.6 GiB resident (`nvidia-smi`) — enough headroom for batch 8 × sequences carrying full option lists on a 12 GB card.

## Bug #3 — the Triton recompilation black hole

One OOD run sat **3 h 12 m at 100 % GPU utilization with zero progress in the log**, CPU 0 %. Textbook signature of kernel-shape churn: the gated-delta-rule kernels compile per input *shape*, and our padded widths were effectively random, so every microbatch triggered a fresh multi-minute JIT compile. Nothing crashed; the run would technically end in days.

Two fixes, both cheap, both mandatory for custom loops:

```python
BUCKET = 128
def left_pad_bucketed(chunks, pad_id):        # 1) pad width up to a 128-token
    width = ((max(c.length for c in chunks) + BUCKET - 1) // BUCKET) * BUCKET
# 2) sort by length WITHIN each random wave (the eval kit's batching trick),
#    so a wave sees ~1-2 shapes instead of ~14 novel ones per epoch
ordered = sorted(wave, key=lambda cg: int(cg[0].length))
```

Afterwards steady-state was 11 s/microbatch and predictable. Lesson generalized: any framework with shape-specializing JIT kernels (Triton, torch.compile, Flash-attn variants) turns ragged batching into a performance cliff. Bound your shapes.

## Act 2 — the teacher that couldn't be beaten (and why we stopped trying)

Before inventing conventions we tried the reasonable version of a custom router: an invented SaaS on-call taxonomy ("OPS-ROUTE v1") with plausible team names, plain-English descriptions, and deliberately counterintuitive house rules (money-looking-like-email goes to `payments_sre`; login-screen crashes go to `auth_identity`, not mobile). Stock model, rules never shown, 80 held-out incidents:

**78/80 = 97.5%** accuracy (chance floor 1/8 = 12.5%), average gold-option confidence 0.95+. On a domain that exists in no pretraining corpus, with house rules we invented ten minutes earlier. Why? Because plausible team *names* plus plain descriptions make routing **reasoning-shaped**: a capable model reconstructs your intent from semantics regardless of what you decided the org chart says. Fine-tuning cannot beat reasoning — so Act 1's lesson generalized: don't tune to improve judgment.

To create real learning headroom we removed exactly one thing: meaning. Enter OPS-ROUTE v2 — the experiment that decides this tutorial.

## Act 3 — Conventions nobody could guess

### Design: knowledge that must be taught

Eight teams with codenames carrying **zero semantic signal**, described only as opaque labels ("kestrel": "squad K"). Gold labels come from three rule families published nowhere except our labels:

1. A hidden **service × region dispatch chart**: five fictional services (`gimbal`, `sprocket`, `thicket`, `lantern`, `cobbler`) × four regions → eight squads, arranged arbitrarily. Nothing about "kestrel" hints it owns `lantern` in `sa-gru`.
2. **Priority keyword overrides**: any ticket tagged `audit` → `verdigris`; else `quarantine` → `onyx`; else `runbook` → `marlin` — overriding the chart regardless of service or region.
3. ~18% of rows carry an override token so both rule layers get supervised.

Anti-shortcut constraint: **train and test brand vocabularies are disjoint** (`ZephyrPay…` never appears at test; test uses `Cloudberry/Tessera/Mothlight`). Memorizing sentences can't transfer; internalizing the chart does. Example row (generated by the downloadable script, deterministic seed):

```
[Quantly] gimbal health checks are flapping between healthy and down in eu-fra Paged at 04:12 UTC.   -> onyx
[ZephyrPay] thicket in sa-gru: latency p99 crossed 8 seconds … A compliance runbook flag was attached. -> marlin
```

### Bug #4 — the dataset that capped itself at 45%

Our first two v2 runs that actually took optimizer steps plateaued at ~46-47.5% val accuracy while train loss fell to zero — textbook overfitting signature, but the plateau sat at a suspicious constant across different epochs and learning rates. It wasn't the model or the optimizer. In our generator, five of six incident templates never mentioned the **region**, yet gold was `CHART[service][region]`. Five-sixths of labels were unknowable from the input text; any model, perfect training or not, caps near 0.45 there. Runs 3 and 4 had converged *optimally* on unlearnable data. The fix is one line per template (`… down in {r}`), the lesson is permanent:

> When accuracy plateaus at a weird fraction of label space, compute the Bayes ceiling of your own dataset before touching hyperparameters. Every field that determines a label must be visible in the input.

### Gate first, train second

With an informative dataset, we verified the gate condition before spending GPU: stock model on the frozen 120-row brand-disjoint test set → **14/120 = 11.7%**, statistically the chance floor, dumping probability onto whichever squad name it liked (confusions into `verdigris` everywhere, avg confidence ~0.12). A task where base ≈ chance AND humans-with-the-rules score 100% is exactly where fine-tuning earns its keep. Only then did we train.

### The run

```bash
python train-router-lora.py --ood --epochs 30 --batch 8 --accum 8 --lr 3e-4 \
    --out lora-ood5     # QLoRA NF4 base, LoRA r=16/a=32 + full-param readout head
```

The request protocol matters as much as the optimizer: batches are built through the model's **own encoder** (`Decider.encoder` → `left_pad_bucketed` → hooked backbone → `readout(hidden[rows, cols])`), i.e. training reuses the shipped inference code path token-for-token so train and eval never drift apart. Supervision is cross-entropy over the 255 option codes at each marker position.

Real log lines (with bug #1 fixed — first time any run of ours ever printed them):

```
step   5/300 loss~1.9670 lr=1.00e-04   58s eta 3414s
step 280/300 loss~0.0000 lr=3.63e-06 3277s eta  234s
FINAL val_loss=0.0000 val_acc=1.0000        # val shares train brands
```

300 optimizer steps, **55 minutes** (wall_s 3303 in `train-log.json`), comfortably inside the card's budget. The held-out slice — same brand pool as training — hit val_acc 1.0 at the first checkpoint (step 100). The number that counts is the probe.

### The result

```
$ python eval-router-probe.py            # stock weights, frozen 120-row test set
OOD ACCURACY: 14/120 = 0.117  (random floor = 1/8 = 0.125)
$ python eval-router-probe.py --adapter lora-ood5/adapter --readout lora-ood5/readout.safetensors
OOD ACCURACY: 120/120 = 1.000  (random floor = 1/8 = 0.125)
  basalt   acc=6/6    avg_gold_conf=1.000
  cinder   acc=16/16  avg_gold_conf=1.000
  ember    acc=9/9    avg_gold_conf=1.000
  harrier  acc=16/16  avg_gold_conf=1.000
  kestrel  acc=20/20  avg_gold_conf=1.000
  marlin   acc=26/26  avg_gold_conf=1.000
  onyx     acc=16/16  avg_gold_conf=1.000
  verdigris acc=11/11 avg_gold_conf=1.000
top confusions: (none)
```

**Chance to perfect.** Every squad at full marks, zero confusions, average confidence in the gold option exactly 1.000 — on brand names ([Tessera], [Cloudberry], [Mothlight]) never once present in training. The adapter learned an org chart that exists nowhere else in the world. That is what you're buying with fine-tuning: **memorized convention, verifiable against held-out facts.**

## What this means for your routing budget

The two halves of this experiment were run by the same script family, on the same GPU, hours apart. Read them together:

- **Your router task is reasoning-shaped if** a competent human reading the ticket *and the team descriptions* can derive the right answer. BANKING77-style intent classification, support triage with meaningful queue names → pick the best pretrained model you can serve, do NOT fine-tune to "improve" it; we measured -0.8 pt for the trouble, calibration included.
- **Your router task is convention-shaped if** the right answer depends on facts a smart newcomer could only learn by being told: codename squads, legacy service ownership matrices, "always page X for audit tags," regional runbook idiosyncrasies. That knowledge will never be in pretraining, no prompt carries all of it reliably, and a 30-minute LoRA on a few hundred examples installs it perfectly — measured here from chance to 100%.
- **The verification ladder is the tutorial.** Every claim above was caught by something before publication: frozen splits + kit-native metrics (act 1), base-at-chance gate (act 3), brand-disjoint test vocab, step-line checks after launch, memory probes when checkpointing looks inert. Zero-step trainers and self-capped datasets both *looked* like completed experiments. They were minutes away from shipping as conclusions.

## Make it yours

Swap in your own org chart in twenty minutes: edit the `TEAMS`, `CHART`, and `KEYWORD_OVERRIDES` tables in `gen_opsroute_dataset.py` (every label-determining field in a template — learn from bug #4), regenerate, re-gate (stock model **must** land near chance before training is meaningful), then run the trainer with the flag set from the log lines above. Keep the three guardrails and your 100% will mean something: frozen test splits, disjoint brand vocab, step-lines-early verification.

Stack pins for reproduction: torch 2.13.0+cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2 on RTX 3080 Ti 12 GB, CUDA driver 595.x. All numbers in this post come from logged runs, not spot checks.


## Where to go next

- New to LoRA mechanics? [LoRA Fine-Tuning: Qwen3.5-4B on 12 GB](../lora-fine-tuning-qwen3.5-4b) builds the adapter math from scratch.
- Want the full pipeline view first? [End-to-End LoRA Pipeline](../end-to-end-lora-pipeline) covers data → train → serve in one pass.
- Before you trust *any* fine-tuned checkpoint — including the 100% one above — measure it the way Act 1 measured this one: [Evaluate a Fine-Tuned LLM](../evaluate-fine-tuned-llm).

*If your fine-tune can't beat the stock model, you haven't failed — you've discovered your task was reasoning-shaped all along. And if it plateaus at a suspicious fraction of perfect, go check the dataset before you touch another hyperparameter.*
