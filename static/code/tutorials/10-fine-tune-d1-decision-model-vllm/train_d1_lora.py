"""Tutorial #10 aiml.training: fine-tune d1-3B (whitelisted decision arch) into
the OPS-ROUTE router, to then be SERVED through /v1/systemone -- the
'trained AND invited' cell of the two-axis map.

Discipline (silent-failure handbook):
- Prompt bytes come from the SAME construction the endpoint will use at serve
  time: falsy state -> content == ask (vLLM 0.33 LiquidStrategy._content),
  options lines '{code} {description}', 'Reply with the option code only.'
- Loss on the answer token ONLY (prompt masked to -100).
- Stage checks print loudly: trainable-param count must be >> 0, loss must be
  finite and falling, adapter must contain tensors matching target modules.

Text tower trains WITHOUT the vision tower: the checkpoint's
text_config is a plain LFM2 config, so we build Lfm2ForCausalLM on it and
load only model.language_model.* weights (keys map 1:1 minus the prefix;
lm_head is tied to embed_tokens).
"""
import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Lfm2ForCausalLM, AutoTokenizer

MODEL_DIR = "/home/agent/aiml-validation/models/d1-3B"
DATA = "/home/agent/tierA/traindata/opsroute2.json"
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "in_proj"]


def ask_block(text, teams):
    """Byte-identical to LiquidStrategy._content for falsy state, choice type:
    '{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only.'
    Lines are '{code} {description or name spaced}' -- our descriptions are
    non-empty ('squad X') so description wins, exactly as at serve time."""
    instructions = "Route this on-call incident to the owning team.\n" + text
    lines = "\n".join(
        f"{chr(65 + i)} {desc}" for i, desc in enumerate(teams.values())
    )
    return f"{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."


def build_pair(tok, row, teams):
    content = ask_block(row["text"], teams)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    code = chr(65 + list(teams.keys()).index(row["team"]))
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ans_ids = tok(code, add_special_tokens=False)["input_ids"]
    assert len(ans_ids) == 1, f"answer code {code} is not a single token"
    ids = prompt_ids + ans_ids
    labels = [-100] * len(prompt_ids) + ans_ids
    return ids, labels, code


def load_text_model():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=False)
    tcfg = cfg.text_config
    model = Lfm2ForCausalLM(tcfg)
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    # KEY MAP (verified by probe, 2026-11-09): checkpoint names the text tower
    # 'model.language_model.X'; native Lfm2ForCausalLM names it 'model.X'.
    # Stripping the prefix to bare 'X' loads ZERO tensors -- silent random init.
    pre = "model.language_model."
    text_sd = {"model." + k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    msd = model.state_dict()
    loaded, skipped = {}, []
    for k, v in text_sd.items():
        if k in msd and msd[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)
    # HANDBOOK GATE: this must be an assert, not a print. 266 text-tower
    # tensors exist in the checkpoint; anything far below = key-map bug.
    assert len(loaded) >= 260, (
        f"only {len(loaded)}/{len(text_sd)} text-tower tensors loaded -- "
        f"key map wrong, model would train from random init. "
        f"skipped sample: {skipped[:5]}")
    model.load_state_dict(loaded, strict=False)
    print(f"[stage] loaded {len(loaded)}/{len(text_sd)} text-tower tensors; "
          f"skipped {len(skipped)} e.g. {skipped[:3]}")
    return model, tcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)  # smoke test knob
    ap.add_argument("--out", default="/home/agent/tierA/d1-opsroute-lora")
    ap.add_argument("--save-first-step", action="store_true",
                    help="save an adapter after step 1 (end-to-end save-path check)")
    args = ap.parse_args()

    d = json.load(open(DATA))
    teams = d["teams"]
    rows = d["train"]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pairs = [build_pair(tok, r, teams) for r in rows]
    lens = [len(p[0]) for p in pairs]
    print(f"[stage] {len(pairs)} train pairs; token len min/med/max = "
          f"{min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    # show one fully decoded pair so the format is auditable
    ids, labels, code = pairs[0]
    print("[audit] PROMPT >>>", repr(tok.decode(ids[:-1]))[-400:])
    print("[audit] ANSWER >>>", repr(code), "loss-token ids", [labels[-1]])

    model, tcfg = load_text_model()
    model.to(torch.bfloat16).to("cuda")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=TARGETS,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train > 1e6, f"only {n_train} trainable params -- adapters did not attach"

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    grad_norm = math.sqrt(len(pairs))  # rms accumulation scale
    model.train()
    losses = []
    step = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(int(math.ceil(args.epochs))):
        for i, (ids, labels, _) in enumerate(pairs):
            dev = model.get_input_embeddings().weight.device
            inp = torch.tensor([ids], device=dev)
            lab = torch.tensor([labels], device=dev)
            out = model(input_ids=inp, labels=lab)
            (out.loss / (args.accum * grad_norm)).backward()
            losses.append(float(out.loss))
            if (i + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if len(losses) >= 8:
                    recent = losses[-args.accum:]
                    print(f"[train] epoch {ep} step {step} "
                          f"loss(last)={losses[-1]:.4f} "
                          f"mean(last{args.accum})={sum(recent)/len(recent):.4f}")
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    assert all(math.isfinite(l) for l in losses), "non-finite loss observed"
    head = sum(losses[:20]) / 20
    tail = sum(losses[-20:]) / 20
    print(f"[result] loss head20={head:.4f} -> tail20={tail:.4f}")
    model.save_pretrained(args.out)
    print("[stage] adapter saved to", args.out)
    files = sorted(os.listdir(args.out))
    print("[stage] adapter files:", files)


if __name__ == "__main__":
    if "--save-first-step" in __import__("sys").argv:
        # end-to-end save path check: 1 step, then save
        os.environ["SAVE_AT_STEP"] = "1"
    main()
