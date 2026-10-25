"""Tutorial #6 LoRA trainer — aiml.training (see the full writeup at
https://aiml.training/docs/tutorials/fine-tune-decision-router/).

LoRA fine-tuning of pplx-decider-flash-v1.2 (base: Qwen3.5-4B, Apache-2.0).
Compute-path fidelity: batches are built with the SAME Encoder / left_pad /
backbone hook / readout as Decider.reference_logits, so identical tensors
flow through training and evaluation. Supervision = cross-entropy over the
readout codes at each answer-marker position.

Two modes:
  (default)        frozen-suite style: BANKING77 + CLINC150+OOS train splits
                   (upstream TRAIN csv/json in --data-dir; never test rows!)
  --ood PATH.json  OPS-ROUTE convention dataset from gen_opsroute_dataset.py

Hard-won traps already fixed here (each cost us a run; see article):
  * optimizer-step gating uses a MICROBATCH counter — gating on the step
    counter itself (step only advances inside its own gate) silently takes
    ZERO optimizer steps while loss.backward() burns GPU for hours.
  * gradient checkpointing must be enabled AFTER get_peft_model() AND the
    module set to .train(): HF gates recomputation on BOTH flags.
  * padded widths are bucket-rounded (BUCKET tokens) inside length-sorted
    waves: unbounded per-batch widths make Triton JIT-recompile every novel
    shape (~minutes each) and the run effectively never ends.
  * bf16 base does not fit backprop on 12 GB — use --quantize (NF4 QLoRA).

Env (RTX 3080 Ti 12 GB, torch 2.13.0+cu130, transformers 5.19.0,
peft 0.21.2, bitsandbytes 0.50.2):
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # if vLLM/registry JIT headers are needed on your box: CPATH trick, see article

Run:
    python train-router-lora.py --ood opsroute2.json --epochs 30 --lr 3e-4 \
        --batch 8 --accum 8 --quantize --out lora-ood5
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

BUCKET = 128  # round padded widths up to this token multiple (Triton JIT trap)


def load_examples_bench(data_dir, limit_bank, limit_clinc, seed):
    """Frozen-suite style examples: request dicts (protocol schema) + gold index.

    data_dir must contain upstream TRAIN splits only:
      banking_train.csv (text,category), banking_categories.json (list),
      clinc_data_full.json ({"train": [[text, label], ...], ...})
    """
    import csv
    data_dir = Path(data_dir)
    ex = []
    labels = json.loads((data_dir / "banking_categories.json").read_text())
    with (data_dir / "banking_train.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["category"] in labels]
    random.Random(seed).shuffle(rows)
    for r in rows[:limit_bank]:
        ex.append(({"state": {}, "questions": {"q1": {
            "type": "choice",
            "instructions": "Classify the banking intent of this user request:\n" + r["text"],
            "criteria": {lbl: lbl for lbl in labels},
        }}}, labels.index(r["category"])))
    obj = json.loads((data_dir / "clinc_data_full.json").read_text())
    tr = obj["train"]
    random.Random(seed + 1).shuffle(tr)
    labs = sorted({p[1] for p in tr} | {"oos"})
    descs = ["out of scope: none of the listed intents" if l == "oos"
             else l.replace("_", " ") for l in labs]
    for text, lab in tr[:limit_clinc]:
        ex.append(({"state": {}, "questions": {"q1": {
            "type": "choice",
            "instructions": "Classify the intent of this user request, or choose out of scope if none applies:\n" + text,
            "criteria": {d: d for d in descs},
        }}}, labs.index(lab)))
    return ex


def load_examples_ood(path, seed):
    """OPS-ROUTE convention dataset: gold lives ONLY in the labels."""
    data = json.load(open(path))
    crit = dict(data["teams"])
    items = list(data["train"])
    random.Random(seed).shuffle(items)
    ex = []
    for r in items:
        ex.append(({"state": {}, "questions": {"q1": {
            "type": "choice",
            "instructions": "Route this on-call incident to the owning team.\n" + r["text"],
            "criteria": crit}}},
            list(crit).index(r["team"])))
    # NOTE: probability keys follow criteria INSERTION order (the protocol
    # encodes options in dict order). Use list(crit).index(...), NOT sorted().
    return ex


def left_pad_bucketed(chunks, pad_id):
    """left_pad() with width rounded up to BUCKET; padding semantics identical."""
    width = ((max(c.length for c in chunks) + BUCKET - 1) // BUCKET) * BUCKET
    ids = torch.full((len(chunks), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(chunks), width), dtype=torch.long)
    for r, c in enumerate(chunks):
        ids[r, width - c.length:] = c.input_ids
        mask[r, width - c.length:] = 1
    inputs = {"input_ids": ids, "attention_mask": mask}
    typed = [c.mm_token_type_ids for c in chunks if c.mm_token_type_ids is not None]
    if typed:
        types = torch.zeros((len(chunks), width), dtype=typed[0].dtype)
        for r, c in enumerate(chunks):
            if c.mm_token_type_ids is not None:
                types[r, width - c.length:] = c.mm_token_type_ids
        inputs["mm_token_type_ids"] = types
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to pplx-decider-flash-v1.2 checkout")
    ap.add_argument("--data-dir", default="./traindata", help="benchmark train-split dir")
    ap.add_argument("--ood", default=None, metavar="JSON",
                    help="OPS-ROUTE dataset path (convention mode instead of benchmarks)")
    ap.add_argument("--bank", type=int, default=3000)
    ap.add_argument("--clinc", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--quantize", action="store_true", help="QLoRA: NF4 4-bit base (fits 12 GB)")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--out", default="lora-out")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.ckpt) / "source" / "src"))
    from pplx_decider.model import Decider  # noqa: E402

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    decider = Decider.from_pretrained(args.ckpt, device="cuda", fast=False, batch_tokens=100_000)
    if args.quantize:
        # bf16 backbone + activations do NOT fit 12 GB for backprop. Swap in
        # the QLoRA base: NF4 + double quant, bf16 compute (tutorial-standard
        # recipe). Re-attach the noncausal mask hook to the NEW instance.
        from transformers import BitsAndBytesConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        from pplx_decider.model import enable_noncausal_full_attention
        del decider.backbone
        torch.cuda.empty_cache()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        decider.backbone = Qwen3_5Model.from_pretrained(
            args.ckpt, quantization_config=bnb, attn_implementation="sdpa",
            device_map="cuda:0")
        enable_noncausal_full_attention(decider.backbone.language_model)
        decider.backbone.requires_grad_(False)
        print("backbone -> NF4 4-bit (QLoRA)", flush=True)
    enc = decider.encoder
    readout = decider.readout
    readout.requires_grad_(True)

    if args.ood:
        examples = load_examples_ood(args.ood, args.seed)
        print(f"OOD mode: {len(examples)} examples from {args.ood}", flush=True)
    else:
        examples = load_examples_bench(args.data_dir, args.bank, args.clinc, args.seed)
    random.shuffle(examples)
    n_val = min(args.n_val, len(examples) // 4)
    val, train = examples[:n_val], examples[n_val:]
    print(f"train={len(train)} val={len(val)}", flush=True)

    # Pre-encode once (CPU) with the frozen protocol encoder; keep only
    # single-marker plans (the ones reference_logits supervises identically).
    t0 = time.time()

    def encode_all(items):
        out = []
        for req, gold in items:
            e = enc.plan(req, None)[0]
            if int(e.markers.numel()) == 1:
                out.append((e, int(gold)))
        return out

    tr_enc = encode_all(train)
    va_enc = encode_all(val)
    print(f"encoded {len(tr_enc)+len(va_enc)} in {time.time()-t0:.1f}s (dropped multi-marker)", flush=True)

    # --- LoRA ---
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=[
            "gate_proj", "up_proj", "down_proj",                          # MLP
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",  # linear attn
            "q_proj", "k_proj", "v_proj", "o_proj",                        # full attn layers
        ],
        bias="none",
    )
    pefted = get_peft_model(decider.backbone, lcfg)
    if not args.no_grad_ckpt:
        # enable AFTER wrapping, then flip to train mode: HF checkpoints only
        # when BOTH module.gradient_checkpointing AND module.training are set
        pefted.get_base_model().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    pefted.train()
    decider.backbone = pefted
    trainable = [p for p in pefted.parameters() if p.requires_grad] + list(readout.parameters())
    n_tr = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_tr/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(tr_enc) / (args.batch * args.accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / warmup) * (
        0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, (s - warmup) / max(1, total_steps - warmup)))))))

    pad_id = enc.pad_id

    def make_batch(chunk_items):
        chunks = [c for c, _ in chunk_items]
        inp = left_pad_bucketed(chunks, pad_id)
        inp = {k: v.to(decider.device) for k, v in inp.items()}
        width = inp["input_ids"].shape[1]
        rows = torch.arange(len(chunks)).to(decider.device)
        cols = torch.tensor([int(c.markers[0]) + (width - c.length) for c in chunks],
                            device=decider.device)
        tgt = torch.tensor([g for _, g in chunk_items], device=decider.device)
        return inp, rows, cols, tgt

    def ce_batch(items):
        inp, rows, cols, tgt = make_batch(items)
        hidden = decider.backbone(**inp, use_cache=False).last_hidden_state
        logits = readout(hidden[rows, cols])
        return torch.nn.functional.cross_entropy(logits.float(), tgt)

    @torch.no_grad()
    def evaluate(items, n=256):
        loss, acc, cnt = 0.0, 0, 0
        for i in range(0, min(n, len(items)), args.batch):
            b = items[i:i + args.batch]
            inp, rows, cols, tgt = make_batch(b)
            hidden = decider.backbone(**inp, use_cache=False).last_hidden_state
            logits = readout(hidden[rows, cols]).float()
            loss += torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
            acc += (logits.argmax(-1) == tgt).sum().item()
            cnt += len(b)
        return loss / cnt, acc / cnt

    log = []
    opt.zero_grad(set_to_none=True)
    step = 0
    mb = 0  # MICROBATCH counter: gating opt.step() on `step` itself deadlocks
            # the accumulator forever (condition can never fire since step only
            # advances inside that branch). Two full runs trained nothing this way.
    t0 = time.time()
    for ep in range(int(math.ceil(args.epochs))):
        random.shuffle(tr_enc)
        # within random effective-batch waves, sort by token length: far fewer
        # distinct padded widths -> fewer Triton compiles, less pad waste,
        # while wave-level shuffling keeps sample order random.
        wave = args.batch * args.accum
        ordered = []
        for ws in range(0, len(tr_enc), wave):
            ordered.extend(sorted(tr_enc[ws:ws + wave], key=lambda cg: int(cg[0].length)))
        for bi in range(0, len(ordered), args.batch):
            batch = ordered[bi:bi + args.batch]
            if len(batch) < args.batch:
                continue
            loss = ce_batch(batch) / args.accum
            loss.backward()
            mb += 1
            if mb % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == total_steps:
                    el = time.time() - t0
                    print(f"step {step}/{total_steps} loss~{loss.item()*args.accum:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} {el:.0f}s eta {total_steps*step**-1*el - el:.0f}s",
                          flush=True)
                    # VERIFY EARLY: first printed step should show LOSS MOVING.
                    # If steps don't print at all, or loss never drops, stop —
                    # you have a plumbing bug, not a training problem.
                    if step % 100 == 0:
                        vl, va = evaluate(va_enc)
                        log.append({"step": step, "val_loss": vl, "val_acc": va})
                        print(f"  val_loss={vl:.4f} val_acc={va:.4f}", flush=True)
        if step >= total_steps:
            break

    vl, va = evaluate(va_enc, n=500)
    print(f"FINAL val_loss={vl:.4f} val_acc={va:.4f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pefted.save_pretrained(str(out / "adapter"))
    from safetensors.torch import save_file as _sf
    _sf({"weight": readout.weight.detach().cpu()}, str(out / "readout.safetensors"))
    (out / "train-log.json").write_text(json.dumps({
        "args": vars(args), "final_val_loss": vl, "final_val_acc": va, "log": log,
        "trainable_params": n_tr, "wall_s": time.time() - t0, "torch": torch.__version__,
    }, indent=2))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
