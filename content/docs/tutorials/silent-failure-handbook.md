---
title: "The Silent Failure Handbook: 10 bugs that made single-GPU LLM runs look successful when they weren't"
slug: "silent-failure-handbook"
description: "Ten real failure modes from our logged RTX 3080 Ti runs — zero-step trainers, checkpointing that never activated, Triton JIT churn, the AutoModel merge no-op, label-token collapse — with the five-minute check that catches each one before you spend hours trusting a dead run."
date: 2026-10-28
lastmod: 2026-10-28
draft: false
weight: 8
type: docs
images: ["og/silent-failure-handbook.png"]
toc: true
author: "K J"
categories: ["Tutorials"]
tags: ["debugging", "fine-tuning", "qlora", "vllm", "12gb", "model-verification"]
prev: docs/tutorials/serve-decision-model-vllm-systemone
sidebar:
  open: true
headerBadge: "Field guide"
headerLead: "Every entry here is a bug we actually shipped-shaped: the run finished, the metrics looked plausible, and the experiment was worthless. None of them crashed. That is what makes them expensive. Each section gives you the symptom as you'll see it, the five-minute check that catches it, and the guard to add so it can never pass silently again."
---

This handbook is the compressed version of everything behind the [verification ladder](../end-to-end-lora-pipeline/) in tutorial #5 and the three silent-bug narratives in [tutorial #6](../fine-tune-decision-router/). It is deliberately cross-cutting: the tutorials are organized by pipeline stage, this one is organized by **failure signature** — what it looks like at 2 a.m. when the GPU is pinned at 100% and something is wrong.

Ordering is by cost. The first three cost us days each because they let a completed run *look* like an experiment.

Two rules make the whole discipline cheap:

1. **Never trust a result whose mechanism wasn't observed.** A loss curve is not evidence that weights updated. Serving output is not evidence that your adapter loaded. An accuracy number is not evidence the task was learnable.
2. **Prefer checks that fail loudly at launch over checks that inform you at the end.** Every guard below runs in the first minute or costs minutes up front, not hours retroactively.

If you only install three guards: print step lines within the first two minutes of any training run (`#1`), assert the attached-LoRA module count after every `PeftModel.from_pretrained` (`#5`), and gate training on a base-model-at-chance measurement (`#7`). Those three alone caught most of what bit us.

---

## 1. The zero-step trainer — GPU pinned, weights frozen, forever

**Symptom.** Training "runs" for hours. `nvidia-smi` shows 100% utilization, power draw is right, memory is right. The loss trace looks like noise around the initial value rather than a decay. When you finally inspect the log file, the structured step log is empty — literally `"log": []`.

**What happened.** Our optimizer-stepping gate was self-referential. The condition for applying gradients required the step counter to hit a residue class, and the counter only incremented *inside* that branch:

```python
# DO NOT COPY — this ran ~4 hours and applied zero optimizer steps
if (step + 1) % accumulation == 0:   # needs step ≡ 7 (mod 8)
    opt.step(); opt.zero_grad()
    step += 1                        # ...but step only advances in here
```

At `step = 0` the gate needs `step ≡ 7 (mod 8)`, which nothing outside the branch can ever produce. Every backward pass computed gradients into `p.grad`; nothing ever consumed them. The model trained for four hours and ended exactly where it started.

Worse: the same pattern had afflicted an earlier run two days before. Its negative conclusion ("our LoRA lost half a point on the benchmark") was really "an identical-weight model, reloaded through a different wrapper, lost half a point" — bf16 tie-breaking wobble flipping near-margin argmaxes. The conclusion happened to survive later reproduction. We did not know that when we wrote it up.

**The five-minute check.** After launching any training run, confirm within two minutes:

```bash
tail -f train.log          # a real step line must appear early
grep -c "loss" train.log   # nonzero within the first few minutes
```

A run with elapsed wall time > per-step cost × 10 and no step line is dead. Kill it and read the gate code.

**The guard.** Maintain a *separate* microbatch counter from the optimizer-step counter, and assert progress mechanically:

```python
micro = 0
for batch in loader:
    loss = fwd_loss(model, batch)
    loss.backward()
    micro += 1
    if micro % accumulation == 0:      # micro advances unconditionally
        opt.step(); opt.zero_grad()
        global_step += 1
    if micro == 2 and global_step == 0:
        raise RuntimeError("optimizer never stepped — inspect gate logic")
```

Also sanity-check that parameters actually move on a 20-step smoke run:

```python
before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
# ... 20 steps ...
moved = sum(1 for n, p in model.named_parameters()
            if p.requires_grad and not torch.equal(p.detach(), before[n]))
assert moved > 0, "no trainable parameter changed"
```

Costs seconds. Would have saved two full GPU-days.

---

## 2. Gradient checkpointing that quietly never turned on

**Symptom.** You enable `gradient_checkpointing=True`, expect a big memory saving, and instead get OOM in backward — or, more insidiously, memory usage looks like you're retaining every activation and nothing errors because the job happens to fit.

**What happened.** HF layers gate checkpointing on *two* conditions: `module.gradient_checkpointing` **and** `module.training`. With PEFT, the order of operations matters: enabling checkpointing before/around wrapping, then leaving the model in eval mode, gives you a config flag set to `True` and a runtime that retains all activations anyway. No warning. Nothing printed.

**The five-minute check.** Don't infer from config; measure. Right after setup, before data touches the model:

```python
model.train()                              # non-negotiable with PEFT
peak_before = torch.cuda.max_memory_allocated() / 2**30
one_backward_step(model, sample_batch)     # tiny shape
print(f"peak after 1 backward: {peak_before:.2f} GiB")
```

Then flip checkpointing off and repeat on the same batch. If both peaks are within a few percent, checkpointing is inert — whatever the config says.

**The guard.** Set it explicitly *after* the PEFT wrap, and assert the internal state:

```python
model = get_peft_model(base, lora_cfg)
model.train()
base.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
assert getattr(base, "gradient_checkpointing", False)
assert base.training, "checkpointing is gated on training mode"
```

On our 12 GB QLoRA run this took peak resident from guaranteed-OOM down to ~10.6 GiB measured by `nvidia-smi` — enough headroom for batch 8 with full option lists in the prompt. Getting it wrong didn't crash; it just meant the job that should have run didn't fit.

Related trap in the same family: the checkpointing flag being inert also defeats the *memory estimate you fed your VRAM budget*. Cross-check plan vs. reality with `torch.cuda.max_memory_allocated()` after a warmup step, not against the calculator's promise — the [VRAM calculator](../../tools/vram-calculator/) publishes its error bars for exactly this reason.

---

## 3. JIT kernel churn — 100% GPU, zero progress, for hours

**Symptom.** The process has been running 3+ hours. GPU utilization reads 100%. CPU is at 0%. The log has not advanced past the first line. There is no error, no traceback, and killing it feels wasteful because "it's working."

**What happened.** Hybrid-GDN (gated-delta-rule) kernels — like Triton kernels generally, and FlashAttention shape-specialized variants — compile per input **shape**, not per batch. Our padded sequence widths were effectively random, so every microbatch triggered a fresh multi-minute kernel compilation. The run would technically have finished in a few days. It was doing real work; it was compiling, not learning.

**The five-minute check.** Before launching, print the shape schedule:

```python
widths = sorted(len(t) for t in tokenized_dataset["input_ids"])
print(f"{len(set(widths))} distinct padded widths across {len(widths)} rows")
```

Hundreds of distinct widths = hundreds of kernel compilations ahead of you. During the run, distinguish compile-stall from training by watching *which* counters move: kernel compile pins the GPU while the step counter sits still.

**The guard.** Bound your shapes. Pad to a small set of buckets (powers-of-two-ish multiples) and sort rows by length within waves so each bucket compiles once:

```python
BUCKETS = [256, 512, 1024, 2048]
def bucketize(n): return next(b for b in BUCKETS if b >= n)
dataset = dataset.map(lambda x: {"bucket": bucketize(len(x["input_ids"]))})
dataset = dataset.sort(column_names=["bucket", "input_ids_length"])  # group shapes
```

After fixing ours, steady state was 11 s/microbatch and predictable, with step lines flowing. Generalize: **any framework with shape-specializing JIT kernels turns ragged batching into a performance cliff. Bound your shapes.** Other places this bites: `torch.compile` with dynamic=False, some MoE routing kernels, and custom CUDA extensions keyed on tensor geometry.

---

## 4. The serving path that ignored your adapter (the silent LoRA no-op)

**Symptom.** You fine-tuned, the loss came down, you serve the model with `enable_lora=True`, and the outputs are... indistinguishable from the base model. No error. No warning. The `/v1/models` endpoint even lists your adapter.

**What happened.** Two independent causes stacked on top of each other, both silent:

1. **vLLM 0.32 + hybrid-GDN LoRA is a known upstream no-op** ([#49354](https://github.com/vllm-project/vllm/issues)). `enable_lora` accepts the adapter, loads it into a registry, and the forward pass never applies it for this architecture family.
2. **The oracle I used to detect it was itself broken**: loading the checkpoint through `AutoModelForCausalLM` resolved to the multimodal wrapper for the VLM-style checkpoint, Peft matched zero modules, applied zero adapters, and reported success. My "adapter does nothing" conclusion was initially an artifact of my own comparison harness.

Fix for (1): don't use `enable_lora` for this arch — `merge_and_unload()` to a dense checkpoint and serve the merged weights (tutorial #5's path). Fix for (2): instantiate the text class explicitly.

**The five-minute check.** Prove adapter effect with a logit delta before testing anything else. Same prompt, adapter off vs on; the max absolute logit difference should be large:

```python
@torch.no_grad()
def logits(model, ids):  return model(ids).logits[0, -1].float()

d = (logits(with_adapter, ids) - logits(base, ids)).abs().max()
print(f"max |Δlogit| = {d:.2f}")   # ours: 11.05. Under ~0.5 = adapter not live
```

We measured `max |Δlogit| = 11.05` on a decisive prompt with the adapter correctly loaded via `Qwen3_5ForCausalLM`. Zero-ish means the adapter isn't applied, regardless of what any library told you.

**The guard.** After every load path — including inside serving scripts — assert the module count:

```python
n = sum(1 for name, _ in model.named_modules() if "lora_A" in name)
assert n == expected, f"attached {n}/{expected} LoRA modules"
```

We now ship this assert in every script in `static/code/`. When the AutoModel path silently attached 0/496, the assert fired at import instead of quietly corrupting a week of conclusions.

---

## 5. The Auto resolver that loads the wrong model class

**Symptom.** Code behaves like your changes did nothing: adapters attach to nothing, weights seem unchanged, generation defaults to the base behavior. Nothing raises, because nothing you asked for actually failed — you just got a different object than you named.

**What happened.** On a checkpoint with a VLM-flavored config, `AutoModelForCausalLM.from_pretrained(...)` resolves to the multimodal wrapper, not the text transformer. Consequences cascade: Peft finds no target modules (see #4), the vision tower eats ~1.3 GiB you needed for KV cache, and prompts route through a path you never intended.

**The five-minute check.** Print the concrete class and the parameter inventory after load:

```python
print(type(model).__name__)                       # Qwen3_5ForCausalLM, not ...ForConditionalGeneration
print(sum(p.numel() for p in model.parameters())/1e9, "B params")
```

Compare against the safetensors index if you suspect missing tensors (#8) — class and tensor inventory together tell you what actually loaded.

**The guard.** Import the concrete class in every published script and treat `Auto*` as prototyping sugar that never ships:

```python
from transformers import Qwen3_5ForCausalLM as TextModel
model = TextModel.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
```

Note the trade-off on constrained hardware: the whitelisted conditional-generation class sometimes carries a vision tower you must shed — `--language-model-only --enforce-eager` under vLLM to stay inside 12 GB. Know which towers are loaded and why.

---

## 6. bf16 full-precision backward on a card that can't hold it

**Symptom.** OOM inside `backward()` — often much later than you'd predict from a forward-pass memory probe. Or it fits during development and dies the moment batch size or sequence length grows.

**What happened.** A bf16 full-precision backbone plus optimizer states plus activations exceeds 12 GB the moment you do real work; the forward pass fits and lies to you. Our bf16 backward on the 4B model needs more than the card has. The fix isn't a knob — it's changing quantization strategy.

**The five-minute check.** Before writing a training loop, compute the floor and compare it to the card:

```text
weights           ≈ 2 bytes × params              (bf16) or 0.5 × params (4-bit NF4)
gradients         ≈ 2 bytes × trainable params
optimizer (AdamW) ≈ 8 bytes × trainable params   (fp32 master + m + v)
activations       ≈ f(batch, seq_len, layers, checkpointing?)   ← the wild card
```

Use the calculator's QLoRA mode for a calibrated starting point and then verify with a measured probe (#2), not by hoping.

**The guard.** On a 12 GB card, default to QLoRA: bitsandbytes NF4 with double quantization and bf16 compute; leave readout heads unquantized (they're small and precision-sensitive); enable checkpointing properly (#2). Our working recipe peaks at ~9.6 GiB with checkpointing active, leaving room for batch and sequence growth. Reserve the ceiling demo (9B) for what it is: demonstrating that it doesn't fit.

A second-order version of this trap: fp8 serving plans. bf16 serving of a 4B under vLLM left us with *negative* KV headroom (−0.13 GiB) — i.e. guaranteed startup failure or pathological preemption. FP8 quantization (Marlin) turned that into +3.4 GiB of KV headroom. Memory planner math is not optional; print the planned KV allocation before launching servers.

---

## 7. Evaluating a task the data cannot answer (the information ceiling)

**Symptom.** Validation accuracy plateaus at a suspicious fraction of the label space — ~1/k, chance-level, or "stuck at 12%" no matter how long you train, and tuning LR does nothing. Loss decreases slightly and flatlines. Everyone suspects hyperparameters. Nobody suspects the dataset.

**What happened.** In one of our routing datasets, templates hid a field that determined the label. Two-thirds of the labels were unrecoverable from the visible input text, so the best achievable accuracy was capped by the *task*, not the training. A perfect optimizer scores maybe 33% and everyone spends a week chasing the remaining 67%.

Odd plateau fractions are the tell. If accuracy stalls at roughly 1/k for a k-way problem, or at the marginal-frequency ceiling of your label distribution, stop training and audit information.

**The five-minute check.** Ask a human-oracle question about 20 misclassified rows: *given exactly the text the model sees, could you produce the gold label?* If a careful reader can't, your model can't. Compute the ceiling explicitly:

```python
from collections import Counter
seen = {}                      # input_text -> labels observed
for row in train: seen.setdefault(row["input"], Counter())[row["label"]] += 1
ceiling = sum(max(c.values()) for c in seen.values()) / len(train)
print(f"information ceiling = {ceiling:.2f}")   # train-set upper bound
```

**The guard.** Make every label-determining field visible in the input template. And before spending GPU at all, gate the experiment on the base model being *at chance*:

```python
base_acc = evaluate(base_model, frozen_test)
assert base_acc < chance + 0.05, "base already solves this — fine-tune adds nothing"
```

That single gate is why tutorial #6's headline number means something: stock pplx-decider-flash scored **14/120 = 11.7%** (chance floor, dumping probability on whichever squad name it liked) and the convention-trained LoRA hit **120/120**. Base-at-chance plus human-with-the-rules-scores-100% is exactly where fine-tuning earns its keep. Conversely, skip the fine-tune where a distilled teacher already tops the benchmark — our honest Act-1 result was losing half a F1 point to the pretrained model on its home turf.

Pair this with a **frozen test set built on disjoint vocabulary** (brand-disjoint classes between train and test) so a high score proves generalization, not memorization.

---

## 8. The checkpoint with no readable readout (silent chance-level classifier)

**Symptom.** Your fine-tuned decision classifier serves cleanly and returns confident-looking probabilities — and scores at chance on everything. Calibration diagnostics read badly from the very first call (`label_mass ≈ 0`). Because generation works and JSON parses, nobody suspects the weights.

**What happened.** One checkpoint we probed ships **no `lm_head` tensor** in its safetensors index. Any attempt to read decisions from first-token logits therefore reads an untrained/default readout. Confidence numbers looked superficially plausible; accuracy was exactly what a coin flip would produce. Our `label_mass ≈ 0` diagnostic caught it before we ran any evaluation, which is the entire point of having that diagnostic.

**The five-minute check.** Inspect the weight manifest before loading anything:

```python
from safetensors import safe_open
idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
keys = set(idx)
assert any("lm_head" in k for k in keys), "no lm_head tensor — logit readout will be garbage"
print(f"{len(keys)} tensors")
```

Run the same scan for the pieces your pipeline assumes: embedding, norm weights, rotary embeddings, router/expert tensors for MoE.

**The guard.** Treat a preflight weight-manifest diff as a build step: compare the checkpoint's tensor names against the class you intend to load (`TextModel(...).state_dict().keys()`), and fail the build on missing entries. Add one sanity eval at chance level: if a supposedly-task-specific model lands at statistical chance on a trivially learnable subset, it isn't the model you think it is. Also verify integrity when downloading release checkpoints: check `sha256` against the published manifest, and investigate mismatches rather than assuming a stale upstream hash until you've reproduced the discrepancy independently (we hit a genuinely stale README hash upstream and confirmed both sides failed identically before moving on).

---

## 9. Benchmarks that leak, drift, or silently swap their splits

**Symptom.** Results improve substantially and reproducibly — then vanish when anyone else runs them, or won't reproduce a month later on "the same" dataset. Sometimes the tell is subtler: a metric moves in a direction your mechanism can't explain.

**What happened.** Three separate ways we saw this close to home:

- **Drift.** Rebuilding eval sets from dataset hubs without pinning gets you different rows next month. Freeze splits and record provenance — our Decision Index rebuild uses upstream sources pinned by SHA-256 in the reproduction kit's manifest.
- **Leakage.** Overlapping train/test content (or near-duplicates) lets a fine-tune memorize instead of learn. Use brand/class-disjoint test vocabularies (#7) and dedupe near-matches explicitly.
- **Wrapper artifacts.** As in #1, a reload through a different wrapper changed results by a small amount with no mechanism behind it — bf16 tie-breaking flips near-margin argmaxes. Small deltas require a mechanism story before they become findings.

**The five-minute check.** Answer three questions out loud for every eval you publish or act on: *which exact rows* (hash them), *which exact metric implementation* (kit-native preferred over a reimplementation), and *what is the smallest change that would have moved the number*. If you can't name a mechanism for a delta, the delta is noise.

**The guard.** Commit eval hygiene to code, not intention:

```python
test_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]
assert test_hash == EXPECTED_HASH, "eval split drifted"
```

Report uncertainty rather than points alone: paired A/B comparisons with Wilson CIs, so "53.3% win rate" travels with "[44.9%, 61.5%] — CI includes 50%, i.e. *not distinguishable from base*". Read the interval, not the midpoint. And use multiple lenses: perplexity, a cheap heuristic judge, and an LLM judge disagreeing is information about *what kind* of improvement you got, not a bug to resolve by picking your favorite.

One more variant of the drift trap, external to your code: **CDN-stale file serving.** Pulling scripts from `raw.githubusercontent.com` minutes after a push served HEAD contents from minutes ago — twice, in one session. Deploy validation scripts by `scp` and verify with `md5sum` on both ends, or wait out the CDN. A "I pushed the fix" + "I ran it" mismatch is a benchmark-drift source that has nothing to do with your code.

---

## 10. Environment and sampler surprises that masquerade as model regressions

**Symptom.** Everything worked yesterday. Today the engine won't start (a compile error mentioning C headers), or sampling produces subtly degenerate text, or the same command now fails on a box that ran it fine last week. You spend time questioning the model, the data, the pins — anything but the environment.

**What happened.** Several distinct mechanisms, all filed under "environment," all burning model-debugging hours:

- **vLLM's registry JIT needs CPython headers** at engine start. On a box with no `python3.11-dev` and no sudo, the fix is pointing the compiler at extracted headers: `export CPATH=/tmp/usr/include:/tmp/usr/include/python3.11` (multiarch wrapper path first). Without it you get a startup-time compile error that looks unrelated to vLLM.
- **Sampler backends can be unavailable in ways that degrade output.** The FlashInfer JIT sampler needs nvcc flags absent from our CUDA 12.4 install. Symptom isn't a clean error; it's flaky/degenerate sampling on a stack that otherwise boots. Fix: force the alternative, `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Version constraints bind in directions docs don't show.** Our vLLM 0.32 requires `transformers>=5.16.1,<5.20.0`, so the site pin stays at 5.19 *by constraint*, not by preference — a naive "upgrade to latest stable" breaks the combo we teach. Similarly, spawning vLLM workers re-import the script, so `LLM(...)` construction must sit under `if __name__ == "__main__":`.
- **Remote-operation footguns.** Process management patterns match the invoking shell's own command line: `pgrep -f train.py` matched the ssh command carrying it, killing our own session and double-launching a trainer. Use bracketed patterns: `pgrep -f '[t]rain.py'`.

**The five-minute check.** Before debugging model behavior, diff the environment against the last known-good:

```bash
pip list --format=freeze | sha256sum
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
env | grep -E "CPATH|VLLM_|CUDA" 
```

Any unexpected change here outranks any hypothesis about your weights.

**The guard.** Publish pins as a contract and note *why* each tight pin exists (this is why every tutorial carries a "version pins and reproduction" block and calls out deliberate drift). Encode required env vars at the top of every script:

```python
import os
os.environ.setdefault("CPATH", "/tmp/usr/include:/tmp/usr/include/python3.11")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
```

And when hardware clocks change underneath you (vBIOS/firmware update on a lab box), **re-baseline every timing figure** — old baselines taken at stock clocks are not comparable to post-unlock runs, and mixing them silently corrupts any throughput claim.

---

## Appendix A — The pre-flight checklist

Print this beside your keyboard. Everything above reduces to these fifteen lines, in order, before you spend real GPU time:

1. Pins installed match the published block verbatim; `pip list` hashed and recorded.
2. Env vars required by the stack are set in the script itself, not in muscle memory.
3. Weight manifest contains every tensor your chosen class expects (`lm_head` included).
4. Concrete model class printed and correct — no `Auto*` in shipping code.
5. Adapter attach-count assert fires non-zero (expected count known ahead of time).
6. Logit-delta probe shows the adapter moves logits (single digits or more, not ~0).
7. Information ceiling computed; no odd 1/k plateaus scheduled to happen.
8. Base model measured at chance on the frozen test set before training is allowed to run.
9. Frozen eval splits hash-checked; test vocabulary disjoint from train.
10. Optimizer takes ≥1 step in the first two minutes, verified from the log, not inferred.
11. Trainable-parameter movement asserted on a short smoke run.
12. Gradient checkpointing measurably active (peak-memory A/B on one batch).
13. Padded widths bucketed; distinct-shape count bounded.
14. Planned serve-time KV headroom printed and positive.
15. Throughput figures come from warm medians on a consistent clock baseline.

Each item is a fifteen-second action that prevents a multi-hour misdiagnosis. The whole checklist costs under ten minutes. The bugs it catches cost days.

## Appendix B — Where each entry came from

These are documented in full, with logs and repro kits, in the tutorials themselves: zero-step trainer and JIT churn (#6), the merge/serving no-op chain and the explicit-class fix (#5, #7), checkpointing activation and QLoRA memory floors (#1, #6), information ceiling and base-at-chance gating (#6), missing-readout detection and calibration diagnostics (#7), serving memory planner and sampler/C-path issues (#2, #5), pinned-split rebuilds and paired-A/B/Wilson methodology (#3). Numbers quoted here are lifted from those logged runs; if you want the raw traces, the downloadable scripts under `/code/tutorials/` reproduce the measurements on the same pins.

Want the estimator that turns Appendix A item 14 into a number instead of a guess? It's free: [VRAM calculator](../../tools/vram-calculator/), calibrated to the same 3080 Ti runs cited above.
