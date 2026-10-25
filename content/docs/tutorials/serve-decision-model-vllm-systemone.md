---
title: "Inside vLLM's /v1/systemone: Decision Serving, and Why Your Fine-Tuned Classifier Wasn't Invited"
slug: "serve-decision-model-vllm-systemone"
prev: docs/tutorials/fine-tune-decision-router
sidebar:
  open: true
description: "vLLM 0.32 ships a native structured-decision endpoint — three questions answered in three output tokens with per-option probabilities and calibration diagnostics. We got it live on a 12 GB card, then tried to ride it with the router LoRA from tutorial #6 and crashed into an architecture gate, a silent merge no-op, and a checkpoint with no lm_head. label_mass caught all of it."
date: 2026-10-25
lastmod: 2026-10-25
draft: false
weight: 7
type: docs
images: ["og/serve-decision-model-vllm-systemone.png"]
toc: true
author: "K J"
categories: ["Tutorials"]
tags: ["vllm", "serving", "decision-models", "qwen3.5-4b", "12gb", "model-verification"]
---

> **Download the full source:** [systemone_demo.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/07-serve-decision-model-vllm-systemone/systemone_demo.py) · [merge_text_lora.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/07-serve-decision-model-vllm-systemone/merge_text_lora.py) · [eval_through_endpoint.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/07-serve-decision-model-vllm-systemone/eval_through_endpoint.py) — runnable versions of everything below.

## The short version

vLLM 0.32 quietly shipped something that looks aimed exactly at readers of [tutorial #6](../fine-tune-decision-router/): a first-class endpoint for *structured decisions*. You POST a state plus typed questions (`choice`, `yes/no`, ordered `score`), and it answers every question by reading the probability of each option's label token — no generation, no parsing, no regex. Our demo call asked three questions at once and the server answered them for **three output tokens total**, returning per-option probabilities and a calibration diagnostic per question.

Then we tried to serve the routing LoRA we trained in tutorial #6 on the same endpoint and it fell apart in three independent ways:

| Attempt | Outcome | Caught by |
|---|---|---|
| Serve upstream Qwen3.5-4B (whitelisted arch) | **Works** — 3 questions, 3 output tokens, label-mass ≈ 0.99 | — |
| Relabel pplx-decider's config to the whitelisted arch | Hard crash at load | HF config-type validation |
| Register our arch in the read-strategy table, serve honestly | Runs, answers — **26/120 = chance level**, `label_mass` **0.0000** | The endpoint itself, before any eval |

The third row is the interesting one: `pplx-decider-flash-v1.2` ships **no `lm_head` at all** (its decisions come from a separate readout head in Perplexity's protocol format), so a naive first-token-logit read scores at chance — and the endpoint's own `label_mass` diagnostic screamed "this model is not deciding anything" without us running a single labeled test. Same species of lesson as tutorial #6: the instrumentation caught what accuracy alone would have missed.

Everything below runs on the same single 12 GB RTX 3080 Ti as every other tutorial on this site (stack pins: torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0, peft 0.21.2).

## What `/v1/systemone` actually is

Not folklore — it is in the wheel. On our box:

```
$ grep -rln systemone .../site-packages/vllm/entrypoints/
.../vllm/entrypoints/generate/structured_decisions/api_router.py
.../vllm/entrypoints/generate/structured_decisions/protocol.py
```

`vllm serve` registers `POST /v1/systemone` automatically (~1,400 lines, Apache-2.0, in `entrypoints/generate/structured_decisions/`). The request schema (`protocol.py`) is small enough to quote in full:

```python
class QuestionSpec(OpenAIBaseModel):
    type: str                    # "choice" | "noul" | "score"
    instructions: Any = ""       # the question text
    criteria: Any = None         # options: dict (choice/noul) or list (score)

class StructuredDecisionRequest(OpenAIBaseModel):
    model: str | None = None
    state: Any                   # what the questions are about: text or JSON
    questions: dict[str, QuestionSpec]   # asked in insertion order
    instructions: str | None     # context placed ahead of the questions
    images / audio / videos: ... # multimodal, for models that perceive
```

Each question type turns `criteria` into lettered options (A–Z) and builds its answer from the **full-vocabulary label probabilities**: `choice` returns the winning option, all option probabilities, and a `confidence`; `noul` returns a yes-probability; `score` returns the *expected* scale position (probability-weighted index). Every answer comes with `diagnostics`:

```python
class QuestionDiagnostics(OpenAIBaseModel):
    label_mass: float    # probability the model put on the labels, over the
                         # FULL vocabulary. Low = most probability went to
                         # tokens that are not labels.
    argmax_is_label: bool
```

That comment — *"a low value means most of the probability went to tokens that are not labels"* — is the whole reason this endpoint can police itself, and we will lean on it hard in act two. Under the hood a read strategy renders the prompt so the label token is the **first and only** generated token (thinking disabled by default), reads one-token logprobs, and softmaxes among the option labels. One forward pass per question; `max_tokens` is structurally 1.

## The gate: which models get invited

Whether your model gets the endpoint is decided once, at startup, in `strategies.py`:

```python
READ_STRATEGIES: dict[str, type[ReadStrategy]] = {
    "ClefOmniForJointSchema": JointSchemaStrategy,
    "Lfm2VlForConditionalGeneration": LiquidStrategy,
    "Qwen3ForCausalLM": NextTokenStrategy,
    "Qwen3_5ForConditionalGeneration": NextTokenStrategy,
    "Qwen3_5MoeForConditionalGeneration": NextTokenStrategy,
    "DiffusionGemmaForBlockDiffusion": DiffusionGemmaCanvasStrategy,
}

def select_read_strategy(model_config, supported_tasks):
    strategy = READ_STRATEGIES.get(model_config.architecture)
    if strategy is None:
        raise ValueError(...)          # -> endpoint stays OFF
    ...
    if strategy.task == "generate" and \
       model_config.logprobs_mode not in LOGPROBS_MODES:
        raise ValueError("Structured decisions need --logprobs-mode "
                         "raw_logprobs or processed_logprobs, not ...")
```

Three gates to pass, all at boot:

1. **Architecture whitelist.** `config.json`'s `architectures[0]` must be a key above. Anything else raises `ValueError` — and here is your first pitfall: `init_structured_decisions_state()` catches that ValueError and logs it at **debug** level (`"/v1/systemone is disabled: %s"`). With default log levels you will never see it. If your decision calls come back 501/NotImplemented, there is no warning in the log telling you why; you have to run with `-v` or read this article.
2. **Task support.** The strategy needs the `generate` task available (true for any normal decoder).
3. **Logprobs mode.** `--logprobs-mode raw_logprobs` (or `processed_logprobs`) on the serve command line. The default silently isn't enough — the readout needs true vocabulary probabilities, not renormalized ones.

What our three checkpoints declare, verified against their `config.json` files:

| Checkpoint | `architectures[0]` | Whitelisted? |
|---|---|---|
| `Qwen/Qwen3.5-4B` upstream | `Qwen3_5ForConditionalGeneration` | ✅ yes |
| text-stripped copy of same | `Qwen3_5ForCausalLM` | ❌ not in table |
| `perplexity-ai/pplx-decider-flash-v1.2` | `Qwen3_5Model` | ❌ not in table |

Note row two: the *same weights*, re-saved through the plain text class, lose eligibility. And note which classes ARE whitelisted: they are the multimodal-capable ones. That coincidence is load-bearing — more in act two.

## Act one: make it work — three questions, three tokens

We served upstream Qwen3.5-4B on the 12 GB card. First attempt OOM'd twice before we understood why: the whitelisted class carries a vision tower (~1.3 GiB of weights we would never use), so fitting it took the two flags that strip it and its capture overhead:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve /models/qwen35-4b \
    --served-model-name q35 \
    --quantization fp8 \
    --logprobs-mode raw_logprobs \
    --language-model-only \
    --enforce-eager \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.92
```

(Two launch attempts OOM'd before these flags landed; together they got the whitelisted VLM-class model onto an 11.6 GiB card. What they buy: `--language-model-only` drops the ~1.3 GiB vision tower we would never use, `--enforce-eager` skips CUDA-graph capture (whose memory spike was what actually tipped us over). Cost of eager: slower decode — irrelevant here, where every answer is exactly one token.)

Then the demo call — a support-ticket router asking three typed questions in one shot:

```json
{ "model": "q35",
  "state": "Ticket #4412: 'I was charged twice for my subscription this month and the second charge went to my old card that I cancelled.'",
  "instructions": "You route support tickets to the correct team.",
  "questions": {
    "route":   { "type": "choice",
                 "instructions": "Which team should handle this ticket?",
                 "criteria": { "billing": "charges, refunds, invoices, payment methods",
                               "tech": "bugs, errors, broken features",
                               "account": "logins, profile settings, cancellations",
                               "sales": "upgrades, plans, enterprise inquiries" } },
    "urgent":  { "type": "noul",
                 "instructions": "Does this customer need money moved back before their next statement?",
                 "criteria": { "true": "a refund or reversal is time-sensitive",
                               "false": "no time pressure on funds" } },
    "severity":{ "type": "score",
                 "instructions": "How severe is this issue?",
                 "criteria": ["cosmetic", "minor", "moderate", "severe"] } } }
```

The response (real, verbatim from our server):

```json
{ "object": "structured_decision", "model": "q35",
  "answers": {
    "route":    { "type": "choice", "choice": "billing",
                  "probabilities": { "billing": 0.99959, "tech": 0.00011,
                                     "account": 0.00030, "sales": 0.0000079 },
                  "confidence": 0.99947 },
    "urgent":   { "type": "noul", "noul": 0.9047,
                  "probabilities": { "yes": 0.9047, "no": 0.0953 },
                  "confidence": 0.8989 },
    "severity": { "type": "score", "score": 2.8844,
                  "legend": { "0": "cosmetic", "1": "minor", "2": "moderate", "3": "severe" },
                  "probabilities": { "0": 0.0019, "1": 0.0017, "2": 0.1063, "3": 0.8900 },
                  "confidence": 0.8880 } },
  "usage":     { "input_tokens": 314, "output_tokens": 3 },
  "diagnostics": {
    "route":    { "label_mass": 0.99989, "argmax_is_label": true },
    "urgent":   { "label_mass": 0.99361, "argmax_is_label": true },
    "severity": { "label_mass": 0.99775, "argmax_is_label": true } } }
```

Stop on `"output_tokens": 3`. Three independent classification decisions — routing, urgency, severity — cost three generated tokens, because each "answer" is exactly one label token whose probability distribution was read instead of sampled. No JSON parsing, no retry-on-malformed-format, no temperature roulette. For comparison, the tutorial #6 harness had to batch through Perplexity's own protocol code with hand-built readouts to get comparable numbers; here it is one POST.

This is what `/v1/chat/completions` wishes it were for classifier-shaped traffic. It is also, notice, *exactly* the request format tutorial #6 trained on — `state`, typed `questions`, named `criteria` labels. vLLM now speaks Decider natively. So the obvious next move was to take our trained OPS-ROUTE adapter from tutorial #6 and serve it here...

## Act two: bring your own router — and get caught lying

Our candidate is the tutorial #6 winner: a QLoRA adapter on `pplx-decider-flash-v1.2` that routes invented-ops incidents to codename squads at **120/120** on a frozen brand-disjoint test set, scored through Perplexity's own readout code. Plan: merge the LoRA, serve it, swap one URL. Three walls.

### Wall 1: the whitelist says no

Serving our checkpoint as-is can't reach the endpoint at all: the boot-time `select_read_strategy` raises for any `architectures[0]` outside the table (Decider ships `Qwen3_5Model`; re-saving through the text class gives `Qwen3_5ForCausalLM`), and `init_structured_decisions_state()` swallows that into a debug log — every call would die with `NotImplementedError: The model does not support the structured decision API` while the rest of the server works normally. It gets worse before it gets better: `Qwen3_5Model` isn't even servable *generatively* on vLLM 0.32 — we tried, and the engine core died mid-load with `ValueError: There is no module or parameter named 'language_model' in Qwen3_5ForEmbedding`. The stripped decoder label routes to an embedding-style class.

### Wall 2: faking eligibility hard-crashes

The lazy fix is a config lie: rewrite `architectures` to the whitelisted `Qwen3_5ForConditionalGeneration` and pray. It dies inside seconds:

```
TypeError: Invalid type of HuggingFace config. Expected type:
  <class 'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5Config'>,
  but found type:
  <class 'transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig'>
```

Now the "multimodal classes only" pattern in the whitelist makes sense: those classes construct a VLM-style config (vision sub-config plus text sub-config) and validate it. Text weights can't satisfy that shape. The gate isn't bureaucratic — it's encoding real coupling between strategy, model class, and config family.

### Wall 3: the honest unlock runs — and scores at chance

The principled route is what upstream should do anyway: register the text architecture in the table. One line into `strategies.py`:

```python
    "Qwen3ForCausalLM": NextTokenStrategy,
+   "Qwen3_5ForCausalLM": NextTokenStrategy,
```

(`NextTokenStrategy` itself is architecture-generic: render chat template, put the label as first token of the reply, read one-token logprobs. Nothing about it needs vision.) This time the server boots clean, `/openapi.json` lists the endpoint, and our call comes back... formatted beautifully. And wrong:

```
$ python eval_through_endpoint.py     # tutorial #6's frozen 120-row test set
SYSTEMONE ACCURACY: 26/120 = 0.2167
label_mass median=0.0000 min=0.0000 max=0.0000
```

(For scale: uniform guessing over eight teams scores ~15/120; the frozen split's largest class has exactly 26 members — 26/120 is the fingerprint of an argmax collapsed onto whichever label the tied-embedding head happens to like.)

Tutorial #6's own harness: **120/120, gold confidence 1.00.** Through vLLM's native endpoint on the same weights: chance. The smoking gun didn't even require the labeled eval — every single response carried `"label_mass": 2.9e-05, "argmax_is_label": false`. In plain English, per the schema's own docs: *almost none of the model's probability is going to any of your labels.* The model being asked isn't deciding; something structural is broken. Check the checkpoint index:

```python
>>> idx = json.load(open("model.safetensors.index.json"))
>>> [k for k in set(idx["weight_map"]) if "lm_head" in k]
[]
```

**No `lm_head`. None.** `pplx-decider` is an encoder-style decision model: its classification probabilities come from a separate readout head fed by non-causal attention over Perplexity's protocol format. Generation-style serving fabricates a weight-tied head from the input embeddings and reads next-token logits off it — which for arbitrary team-name strings gives you uniform noise dressed in a valid JSON envelope. `label_mass ≈ 0` was reporting precisely this all along.

So the endpoint did the thing tutorial #6 taught us to demand of any measurement: it exposed a plausible-looking-but-empty result without needing ground truth. Two numbers worth keeping in your head as calibration tripwires: a healthy decision gets `label_mass` ~0.99; a structural failure gets ~0. Anything in between ("0.3? weird but maybe my prompt?") means investigate before you ship.

### Bonus wall: the silent-noop merge strikes again

Getting to wall 3 required merging our LoRA first — and our first merge was itself a defect. Loading the hybrid model via `AutoModelForCausalLM` and wrapping with `PeftModel.from_pretrained` printed a flood we initially skimmed:

```
UserWarning: Found missing adapter keys while loading the checkpoint:
['base_model.model.model.layers.0.linear_attn.out_proj.lora_A...', ... 496 more]
```

Read plainly: peft attached **zero of 496** adapter tensors. Wrong wrapper class meant module names didn't line up; `merge_and_unload()` dutifully merged nothing and saved a pristine un-fine-tuned model. Same disease as tutorial #5's vLLM-LoRA no-op — a pipeline stage that quietly does nothing while everything downstream looks healthy. The fix is the explicit class (the pitfall our GPU notes already flagged for this model family) **plus an assertion that refuses to proceed on zero attachments**:

```python
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
m = Qwen3_5ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
pm = PeftModel.from_pretrained(m, ADAPTER)
n_lora = sum(1 for n, _ in pm.named_parameters() if "lora_" in n)
# 24 linear-attn layers x 8 targets x 2 + 8 self-attn layers x 7 targets x 2 = 496
assert n_lora == 496, f"MERGE WOULD BE NO-OP: {n_lora} lora params attached"
merged = pm.merge_and_unload()
```

With the right class: exactly 496/496. (Compute your expected count from the adapter's `target_modules` × layer types before writing the assert — this hybrid stack trains `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}` plus MLP projections on GDN layers and `self_attn.{q,k,v,o}_proj` plus MLP on attention layers.) Worth being precise about our own trail: the very first probes through the endpoint (before we caught the merge bug) scored identically badly on the **no-op** merge — wrong team, `label_mass` ≈ 2.9e-5 — and looked indistinguishable from the final 26/120 on the *correct* merge. Two independent defects, same signature; we ruled the merge out only by fixing it properly (496/496) and re-scoring. Both matter because they fail differently: wall 3 you catch from diagnostics, wall 4 you must catch before eval, since accuracy on an unadapted model still looks *plausible* when the base model partially knows the task.

## What it would take to invite your decoder

If you want a checkpoint of yours on this endpoint, work down this list — each item is a wall we hit or verified:

1. **`architectures[0]` must be a real key in `READ_STRATEGIES`.** If your model's class genuinely fits what the strategy does (first-token label readout via the chat template), adding one dict entry is honest and small — upstream review permitting. If your model needs something different (a second scoring path?), write a strategy; do not lie about the architecture, because config validation (wall 2) will reject the lie anyway, and worse lies that *do* load score like ours did.
2. **You need a real generation head.** Any first-token-logits readout silently falls back to tied embeddings when `lm_head` is absent, and nothing raises. Before trusting results: inspect your safetensors index for `"lm_head"` keys. Ours returned `[]`, and everything downstream was fiction from that point.
3. **Serve flags are load-bearing:** `--logprobs-mode raw_logprobs` (default fails gate 3 at boot), and budget memory for whatever baggage your whitelisted class drags along (our +1.3 GiB vision tower needed `--language-model-only --enforce-eager`).
4. **Train/serve format match still applies.** Our tutorial #6 rows trained through Perplexity's protocol encoder — label probabilities computed by their readout over non-causal attention. A serving stack that re-renders your state+questions through the *chat template* is asking a different model a different question. When in doubt, evaluate through the exact path production uses; that single discipline is what turned all three walls from mysteries into two-line diagnoses.

And the contribution-shaped version of this article's finding: `NextTokenStrategy` is nearly architecture-agnostic — the cleanest upstream PR for Qwen3.5 text models is the whitelist line plus tests proving an `lm_head`-bearing text model reproduces offline scores. Everything else in the pipeline worked unmodified once eligibility was resolved honestly.

## Verification kit: five checks before you trust a decision API

The transferable part. For any decision-serving setup — this endpoint or somebody else's:

| Check | Command / signal | Catches |
|---|---|---|
| Endpoint actually registered | `curl :8123/openapi.json \| grep systemone` | debug-level silent disable |
| Logprobs mode | boot args contain `--logprobs-mode raw_logprobs` | boot-time ValueError → disabled |
| Head exists | safetensors index contains `lm_head.*` | fabricated tied-head scoring at chance |
| LoRA attached | assert count of `lora_` params == expected | silent no-op merges |
| Model is deciding | `label_mass` ≈ 1.0 on smoke rows; `argmax_is_label: true` | everything structural, pre-gold |

Run them top-down: cheap static checks first, then the diagnostics tripwire, and only then spend a labeled eval set. Each wall we hit today had a detector strictly cheaper than accuracy.

## Make it yours

Grab `systemone_demo.py` and point it at any instruct model that passes the gate table (plain `Qwen3ForCausalLM` checkpoints qualify out of the box): define one `choice` question with your own criteria dict, fire it at ten rows by hand, and watch the two numbers that matter — winning probability *and* `label_mass`. Put `label_mass < 0.5` behind an alert in anything you serve: it is the cheapest "your model broke" alarm in this entire stack. If your routing task resembles tutorial #6's tribal-knowledge problem, train there, score offline against frozen data, and only graduate to this endpoint once your checkpoint has a genuine head and an eligible architecture.

## Where to go next

- The training side of this story: [Teach a Decision Model Its Tribal Knowledge](../fine-tune-decision-router/) — where the 120/120 router and its readout machinery come from.
- Merge mechanics under vLLM, including the original silent no-op: [End to End: Train, Verify, Merge, Serve](../end-to-end-lora-pipeline/).
- Baseline serving anatomy if `/v1/systemone` isn't available on your version: [Serve a Fine-Tuned LLM with vLLM](../serve-fine-tuned-llm-vllm/).

*If the diagnostics don't agree with your eval set, believe the diagnostics — they were right before your eval set finished running.*
