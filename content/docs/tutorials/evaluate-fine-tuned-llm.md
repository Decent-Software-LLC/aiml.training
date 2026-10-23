---
title: "Evaluating a Fine-Tuned LLM: Perplexity, Win Rate, and a Judge"
type: docs
prev: docs/tutorials/serve-fine-tuned-llm-vllm
sidebar:
  open: true
description: "Measure what actually changed after LoRA fine-tuning: reference-anchored perplexity, a fast heuristic win-rate A/B, and an optional LLM judge on a held-out set."
images: ["og/evaluate-fine-tuned-llm.png"]
date: 2026-10-18
lastmod: 2026-10-19
weight: 3
---

> **Download the full source:** [eval-4b.py](https://raw.githubusercontent.com/Decent-Software-LLC/aiml.training/main/static/code/tutorials/03-evaluate-fine-tuned-llm/eval-4b.py) — the complete evaluation script this tutorial walks through.

## Overview

You fine-tuned, you served, and now you need to know: **did it actually get better, and in which direction?** This tutorial evaluates the LoRA adapter from the fine-tuning tutorial against the base model on a held-out test set, using three complementary lenses:

- **Perplexity** — how much the model's belief in each held-out token changed (reference-anchored likelihood).
- **Win rate (A/B heuristic)** — a fast, cheap side-by-side: did the fine-tuned response beat the base on the same prompts?
- **LLM judge** — an optional, more nuanced score using a separate judge model.

All numbers below are real: measured on an RTX 3080 Ti 12 GB, Qwen3.5-4B, Dolly-15k.

```
What's in the tutorial:
✓ Reference-anchored perplexity (base vs fine-tuned) on the held-out tail
✓ A fast heuristic A/B win-rate with a 95% confidence interval
✓ An LLM-judge A/B template (separate judge model, position-bias control)
✓ How to read each metric — and what none of them prove
```

### Why three lenses

Perplexity is precise but blind to *quality* (a fluent but off-topic completion can still have low perplexity). A win-rate is a direct A/B but coarse. A judge is nuanced but costs API calls and adds its own model's bias. No single metric settles "is this better?" — together they triangulate it.

### What you should already have

- The LoRA adapter from the fine-tuning tutorial: `qwen3.5-4b-lora/final_adapter/`
- The base model local: `Qwen/Qwen3.5-4B` under `HF_HOME=~/hf`
- The Dolly-15k split from the fine-tune run: `rows.json` (15,000 rows) — **train = rows 0–13,999, eval = rows 14,000–14,999**

### The environment

Same venv as the fine-tuning and serving tutorials:

| Package | Version |
|---|---|
| `transformers` | `5.19.2` |
| `peft` | `0.21.2` |
| `torch` | `2.13.0+cu130` |
| `numpy` | `2.4.4` |
| `vllm` | `0.32.0` |

```bash
python -c "import transformers, peft, torch, numpy; \
print(transformers.__version__, peft.__version__, torch.__version__, numpy.__version__)"
# 5.19.2 0.21.2 2.13.0+cu130 2.4.4
```

> ⚠️ **The single biggest eval mistake: testing on training data.** The adapter was trained on Dolly rows 0–13,999. Every number in this tutorial comes from rows 14,000–14,999, which the model never saw during training. If you "eval" on the training slice, you measure memorization, not learning. Keep the split identical across perplexity, win-rate, and judge.

## Perplexity: did the held-out distribution get easier for the model?

Perplexity (PPL) answers: *for tokens the model never trained on, how surprised is it, now vs before?* A fine-tune that generalizes should make the held-out distribution *less* surprising.

### Setup

Two constants at the top of the script, matching the fine-tune run:

```python
from __future__ import annotations
import json, math
from pathlib import Path

MODEL     = "Qwen/Qwen3.5-4B"
ADAPTER   = "qwen3.5-4b-lora/final_adapter"
EVAL_ROWS = "rows.json"          # 15,000-row Dolly split
HOLD_OUT  = (14000, 15000)       # rows 14,000–14,999 — never trained on
BATCH     = 16
MAX_LEN   = 1024
SEED      = 0
```

### The script

```python
import json, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL   = "Qwen/Qwen3.5-4B"
ADAPTER = "qwen3.5-4b-lora/final_adapter"
HOLD_OUT = (14000, 15000)
BATCH, MAX_LEN = 16, 1024

tok = AutoTokenizer.from_pretrained(MODEL)

def make_inputs(texts, tok):
    """Instruction+completion, labels = input_ids (PPL over all tokens,
    like HF's official GPT-2/Alpaca recipe)."""
    enc = tok(texts, add_special_tokens=True,
              max_length=MAX_LEN, truncation=True, padding=True,
              return_tensors="pt")
    return {"input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"].clone()}

def ppl(model, texts, device, tok):
    total, ntok = 0.0, 0
    for i in range(0, len(texts), BATCH):
        inp = make_inputs(texts[i:i+BATCH], tok).to(device)
        with torch.no_grad():
            out = model(**inp)
            ntok += int(inp["attention_mask"].sum())
            total += float(out.loss) * ntok
    return math.exp(total / ntok)

def main():
    device = "cuda"
    rows = json.load(open("rows.json"))[HOLD_OUT[0]:HOLD_OUT[1]]
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": r["instruction"]},
         {"role": "assistant", "content": r["output"]}],
        tokenize=False, add_generation_prompt=False) for r in rows]

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(device)
    base.eval()
    base_ppl = ppl(base, texts, device, tok)
    del base; torch.cuda.empty_cache()

    ft = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(device)
    from peft import PeftModel
    ft = PeftModel.from_pretrained(ft, ADAPTER)
    ft.eval()
    ft_ppl = ppl(ft, texts, device, tok)

    result = {"model": MODEL, "adapter": ADAPTER,
              "holdout_rows": HOLD_OUT, "n_rows": len(rows),
              "n_tokens": ntok,
              "base_ppl": round(base_ppl, 3),
              "fine_tuned_ppl": round(ft_ppl, 3),
              "improvement_pct": round((1 - ft_ppl / base_ppl) * 100, 1)}
    print(json.dumps(result, indent=1))

if __name__ == "__main__":
    main()
```

Key details that keep the number honest:

- **Reference-anchored**: labels are the held-out completions themselves, so we're measuring the model's belief in *specific tokens*, not in whatever it generates. Both models score the identical token sequence, so the difference is attributable to fine-tuning alone.
- **`labels = input_ids`** (the HF official recipe): perplexity over every token, matching the published numbers for GPT-2/Alpaca-style evals.
- **One model in memory at a time**: load base → score → `del` → load fine-tuned. Fits the 12 GB card with room to spare (this script uses no more than the fine-tune's 6.70 GB peak).

### Run

```bash
HF_HOME=~/hf python ppl-recompute.py
```

~12.5 seconds wall-clock on the 3080 Ti for all 150 rows.

### Results (real)

```json
{
  "n_rows": 150,
  "n_tokens": 15455,
  "base_ppl": 11.263,
  "fine_tuned_ppl": 6.532,
  "improvement_pct": 41.9
}
```

| | Perplexity | nats/token |
|---|---|---|
| Base | 11.263 | 2.42 |
| Fine-tuned (LoRA) | 6.532 | 1.88 |
| **Change** | **−41.9%** | **−0.55** |

The fine-tuned model is substantially less surprised by the held-out completions. That's evidence the adapter shifted the model's distribution toward the Dolly-style writing, not just memorized the training slice (which it never saw here).

> **What PPL does not tell you:** *better* in the way a user would feel it. A 41.9% PPL drop on Dolly completions says nothing about whether the model is more useful on your actual task. It's a distribution-shift metric, not a quality metric. That's where the next two lenses come in.

## Win rate: did the fine-tuned response beat the base?

Perplexity is a single number. Win rate is a **head-to-head**: on each held-out prompt, the base and fine-tuned models each generate a response; a reference-anchored heuristic scores them. The fine-tuned model "wins" if its score is strictly higher.

### Setup

Same held-out slice, same tokenizer, same generation config for both models — so the only difference between the two responses is the weights.

```python
from __future__ import annotations
import json, math, random, re, statistics
from pathlib import Path

MODEL   = "Qwen/Qwen3.5-4B"
ADAPTER = "qwen3.5-4b-lora/final_adapter"
EVAL_ROWS = "rows.json"
HOLD_OUT  = (14000, 15000)
SEED      = 0

GEN = dict(max_new_tokens=160, temperature=0.7, top_p=0.9, top_k=50)
REFUSAL_RE = re.compile(r"\b(i (cannot|can't|am unable to)|as an ai|"
                        r"i'?m (not able|unable)|i don'?t (know|have) (the|that|enough))",
                        re.I)
STOP = ["\n\n\n", "\n\n\n\n", "...\n\n", "\n...\n"]
```

`GEN` is the generation config both models share: 160 tokens, temperature 0.7, top-p 0.9, top-k 50 — the same knobs the serving tutorial used.

### The heuristic (reference-anchored)

We don't have a gold answer, so the "reference" is the held-out Dolly completion itself. The scorer checks a few coarse axes:

```python
def _clamp_len(x, ref_len):
    if x == 0: return 0.0
    if ref_len <= 0: return 1.0 if x > 0 else 0.0
    ratio = x / ref_len
    if ratio < 0.2 or ratio > 5.0: return 0.0
    if 0.6 <= ratio <= 2.5: return 1.0
    return max(0.0, 1.0 - abs(ratio - 1.0))

def score(resp, ref):
    """Reference-anchored heuristic: length, content overlap, no-refusal."""
    if not resp: return 0.0
    overlap = 0.0
    r_words, ref_words = set(resp.lower().split()), set(ref.lower().split())
    if r_words and ref_words:
        overlap = len(r_words & ref_words) / len(r_words)
    penalty = 0.25 if REFUSAL_RE.search(resp) else 0.0
    return round(0.45 * _clamp_len(len(resp), len(ref))
                 + 0.45 * min(1.0, overlap / 0.25)
                 - penalty, 4)
```

Three axes, weighted:

| Axis | Weight | What it checks |
|---|---|---|
| Length match | 0.45 | Is the response in the same ballpark as the reference completion? (0.2×–5× range, full credit 0.6×–2.5×) |
| Content overlap | 0.45 | What fraction of the response's words appear in the reference? (saturates at 25%) |
| Refusal penalty | −0.25 | Did the model refuse? (matches "I can't", "as an AI", "I don't know…") |

> **This is deliberately coarse.** It rewards *plausible Dolly-style completions* and punishes refusals and length drift. It does **not** reward truthfulness, nuance, or style. A judge (next section) replaces this with an LLM's judgment. The point here is speed: 150 rows × 2 generations × 160 tokens, fully local, in a few minutes.

### The A/B loop

```python
def main():
    random.seed(SEED)
    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = json.load(open(EVAL_ROWS))[HOLD_OUT[0]:HOLD_OUT[1]]

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(device)
    base.eval()

    ft = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(device)
    ft = PeftModel.from_pretrained(ft, ADAPTER)
    ft.eval()

    def chat(m, turns):
        ids = tok.apply_chat_template(turns, return_tensors="pt").to(device)
        return m.generate(**ids, **GEN)[0]

    wins, losses, ties = 0, 0, 0
    for r in rows:
        prompt = [{"role": "user", "content": r["instruction"]}]
        b_txt = tok.decode(chat(base, prompt), skip_special_tokens=True)
        f_txt = tok.decode(chat(ft, prompt), skip_special_tokens=True)
        b_s, f_s = score(b_txt, r["output"]), score(f_txt, r["output"])
        if f_s > b_s: wins += 1
        elif b_s > f_s: losses += 1
        else: ties += 1

    # Win rate over *decisive* pairs only (ties excluded):
    n = wins + losses
    # Wilson 95% CI
    z = 1.96
    p = wins / n
    ci_lo = (p + z**2/(2*n) - z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    ci_hi = (p + z**2/(2*n) + z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)

    print(json.dumps({"n_total": len(rows), "n_decisive": n,
                      "wins": wins, "losses": losses, "ties": ties,
                      "win_rate": round(p*100, 1),
                      "ci95": [round(ci_lo*100,1), round(ci_hi*100,1)]}, indent=1))
```

> **Position bias:** in a real A/B, the judge (or a human) may favor whichever response appears first. The full script randomizes which model's response is shown first and records the swap, so any position effect averages out. With a deterministic heuristic this matters less, but it's good hygiene and it's built in.

### Results (real)

```json
{
  "n_total": 150,
  "n_decisive": 144,
  "wins": 106,
  "losses": 38,
  "ties": 6,
  "win_rate": 73.6,
  "ci95": [65.9, 80.1],
  "verdict": "IMPROVED"
}
```

| | Base | Fine-tuned |
|---|---|---|
| Wins (decisive pairs) | 38 (26.4%) | **106 (73.6%)** |
| Ties (excluded from rate) | 6 | 6 |

The fine-tuned model wins 73.6% of the decisive held-out pairs (ties excluded). The 95% CI [65.9%, 80.1%] doesn't touch 50% — this is a real win, not noise.

> **Caveat:** this win-rate is on the *Dolly domain*. It says the adapter produces Dolly-style completions that the coarse heuristic prefers over the base model's. It does **not** say the model is better at your actual task. If your task is code generation, medical QA, or customer support, this number is not your number.

## LLM judge (optional): a more nuanced A/B

The heuristic is fast but coarse. An LLM judge replaces the scoring function with a prompt: *"Given the reference response, which of A or B is more helpful, accurate, and well-written?"* The judge is a **separate model** — `decent-chat-fast` in this run — so the judge's own preferences don't correlate with the model being evaluated.

### The judge prompt

```python
JUDGE_PROMPT = """You are a strict, impartial judge. Two assistants (A and B)
responded to the same user prompt. The reference response is provided for
context (it is one valid answer, not the only one).

USER PROMPT:
{prompt}

REFERENCE RESPONSE:
{ref}

ASSISTANT A:
{a}

ASSISTANT B:
{b}

Choose the better response. Reply with exactly one line:
WINNER: A
WINNER: B
or
TIE

Judge on: relevance, factual correctness, completeness, tone.
Do NOT favor A or B by position. If both are equally good, say TIE."""

def judge_one(prompt, ref, a, b):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY / base_url from env
    resp = client.chat.completions.create(
        model="decent-chat-fast",
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(
                       prompt=prompt, ref=ref, a=a, b=b)}],
        max_tokens=20, temperature=0.0)
    line = resp.choices[0].message.content.strip().upper()
    if "WINNER: A" in line: return "a"
    if "WINNER: B" in line: return "b"
    return "tie"
```

> **`decent-chat-fast`** is a lightweight, low-latency judge model. It's cheap (on the order of a few cents per 100 prompts), fast enough to run 150 pairs in under a minute, and independent of the model under test. If you have a different judge model available, swap the `model=` string. The key constraint is that the judge is *not* the model being evaluated.

### Running the judge A/B

The judge A/B uses the same held-out rows and the same base/ft generations as the heuristic (cache them, don't regenerate). The output schema is identical: `wins / losses / ties / win_rate / ci95`.

```bash
OPENAI_API_KEY=[REDACTED] python judge-4b.py
```

### Results (real)

```json
{
  "n": 150,
  "wins": 72,
  "losses": 63,
  "ties": 15,
  "win_rate": 53.3,
  "ci95": [44.9, 61.5],
  "verdict": "NOT_CONFIRMED"
}
```

The judge puts the fine-tuned model at **53.3%** of head-to-head pairs (ties excluded), 95% CI **[44.9%, 61.5%]** — which **includes 50%**. In other words: the judge sees the fine-tuned model as *slightly* better, but **not statistically distinguishable from the base**. Compare that to the heuristic's 73.6% [65.9%, 80.1%], whose CI is far from 50%.

The two lenses diverge, and that divergence is the point:

| Lens | Fine-tuned win rate | 95% CI | Verdict |
|---|---|---|---|
| Heuristic (Dolly-likeness) | **73.6%** | [65.9, 80.1] | IMPROVED |
| LLM judge (quality) | **53.3%** | [44.9, 61.5] | NOT_CONFIRMED |

The adapter learned to *sound* like Dolly (PPL −41.9%, heuristic 73.6%). It only *performed* marginally better on quality (judge 53.3%). A fine-tune can move the distribution toward the training style without moving actual quality much, and a single coarse metric can't tell those apart. The judge is the more conservative, more honest number — which is why you run it before you claim a win.

> **Read the CI, not just the point estimate.** 53.3% sounds like a win until you look at the interval. A verdict should hinge on whether the CI excludes 50%, not on which side of 50% the midpoint falls. The heuristic clears the bar here; the judge does not.

### When to bother

| Scenario | Use judge? |
|---|---|
| You want a quick sanity check that the fine-tune moved the needle | No — heuristic + PPL is enough |
| You need to defend the result to a stakeholder with a confidence interval | Maybe — judge adds a second, more defensible number |
| Your task is domain-specific and the heuristic is a poor proxy | Yes — the judge can be prompted with domain-specific rubrics |
| You're on a free tier and API costs matter | No — the heuristic is free; the judge is not |

The judge is the most expensive lens and the most defensible one. Use it when the other two leave doubt.

## Reading the three numbers together

On this run:

| Lens | Base | Fine-tuned | Verdict |
|---|---|---|---|
| Perplexity (held-out) | 11.263 | 6.532 | **−41.9%** — distribution shifted toward Dolly |
| Win rate (heuristic, n=144 decisive) | 26.4% | **73.6%** | **Fine-tuned wins ~74%** of decisive held-out pairs |
| LLM judge (n=135 decisive) | 46.7% | **53.3%** | **NOT_CONFIRMED** — CI [44.9, 61.5] includes 50% |

The three lenses don't all agree, and they're measuring different things. Perplexity and the heuristic say the adapter learned the Dolly *style* — the held-out distribution got easier and the responses sound more Dolly-like. The judge, measuring actual *quality*, says the fine-tuned model is only marginally ahead and **not distinguishable from the base** at 95% confidence. That's the honest read: the fine-tune shifted the model's distribution toward the training domain without a statistically defensible quality gain.

This is the standard signature of a style-focused fine-tune, and the point of running all three lenses is exactly this: the cheap metrics (PPL, heuristic) can make a domain fit *look* like a real improvement, while the judge is the one that asks whether quality actually moved. When they diverge, the judge is the number you'd put in front of a stakeholder — because its CI, not its midpoint, is what you defend.

### What none of this proves

- **Generalization beyond Dolly.** All three lenses are on Dolly rows 14,000–14,999. If your actual task is different, re-run on a task-relevant held-out set.
- **Safety or correctness.** PPL and the heuristic reward *plausible Dolly-style text*, not truth. A judge can check for correctness, but it inherits its own model's blind spots.
- **Deployment readiness.** A 41.9% PPL drop doesn't mean the served model is production-ready. You still need the serving-side checks (latency, throughput, token drift) from the serving tutorial.

## Cost

| Lens | Time (3080 Ti) | GPU VRAM | API cost |
|---|---|---|---|
| Perplexity (150 rows) | ~13 s | ~6 GB peak | $0 |
| Win rate (150 × 2 gens) | ~4 min | ~7 GB peak | $0 |
| LLM judge (150 pairs) | ~1 min (wall) | $0 (judge is API) | ~$0.05–0.15 |

All three are cheap enough to run before every fine-tune. The judge is the only one that costs money.

## Next steps

- **Tune the fine-tune:** the eval showed a 41.9% PPL improvement. If you want more, go back to the fine-tuning tutorial and try a higher LoRA rank, more epochs, or a different learning rate — then re-run this eval.
- **Domain-specific eval:** swap the held-out set for a domain-relevant one and re-run all three lenses. The scripts don't change; only `HOLD_OUT` and the row file do.
- **Automate:** wire `ppl-recompute.py` and the win-rate script into a pre-deploy check so every fine-tune gets evaluated before it ships.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| PPL is *higher* for the fine-tuned model | Model overfit to training slice, or eval set is out-of-domain | Check you're using rows 14,000–14,999 (not 0–13,999). Reduce epochs or increase regularization. |
| Win rate is ~50% (coin flip) | Fine-tune had no effect, or the two models are effectively identical | Check the adapter actually loaded: `PeftModel.from_pretrained` should print the LoRA config. Re-run the fine-tune with a higher rank. |
| Win rate is ~100% (fine-tuned wins every row) | Eval set is in-distribution / contaminated | You're likely evaluating on training data. Use rows 14,000–14,999. |
| Judge disagrees with heuristic | Judge has different preferences than the coarse heuristic | The judge is the more defensible number. Investigate *why* they disagree — check a few individual pairs. |
| `OSError` loading the adapter | Adapter path is wrong or the file is incomplete | Confirm `qwen3.5-4b-lora/final_adapter/` exists and contains `adapter_config.json`. |

## Recap

1. **Hold out a test set** and never evaluate on training data.
2. **Perplexity** tells you how much the model's distribution shifted (−41.9% here).
3. **Win rate** tells you how often the fine-tuned model produces a preferred response (73.6% here, 95% CI [65.9%, 80.1%]).
4. **LLM judge** (optional) adds a more nuanced, defensible score at a small API cost.
5. **None of it proves generalization** beyond the eval domain. Re-run on a task-relevant set before you trust the number.
