"""Tutorial #6 routing probe — aiml.training (see the full writeup at
https://aiml.training/docs/tutorials/fine-tune-decision-router/).

Runs the OPS-ROUTE frozen TEST split through pplx-decider-flash-v1.2, either
stock or with a trained LoRA adapter + readout head, and prints kit-style
accuracy + average gold-option confidence per squad, plus top confusions.

Printed numbers in the article come from exactly this script:
    stock:              OOD ACCURACY: 14/120 = 0.117
    --adapter lora-ood5: OOD ACCURACY: 120/120 = 1.000

Run (after gen_opsroute_dataset.py produced opsroute2.json):
    python eval-router-probe.py --ckpt /path/to/pplx-decider-flash-v1.2
    python eval-router-probe.py --ckpt ... --adapter lora-ood5/adapter \
        --readout lora-ood5/readout.safetensors
"""
import argparse
import json
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="opsroute2.json")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (train-router-lora.py output)")
    ap.add_argument("--readout", default=None, help="trained readout .safetensors (train alongside --adapter!)")
    ap.add_argument("--out", default="probe-results.json")
    args = ap.parse_args()

    sys.path.insert(0, str(args.ckpt) + "/source/src")
    from pplx_decider import Decider
    import time

    data = json.load(open(args.data))
    crit = dict(data["teams"])
    rows = [(r["team"], {"state": {}, "questions": {"q1": {
        "type": "choice",
        "instructions": "Route this on-call incident to the owning team.\n" + r["text"],
        "criteria": crit}}}) for r in data["test"]]

    d = Decider.from_pretrained(args.ckpt, device="cuda", fast=False, batch_tokens=4096)

    if args.adapter:
        # IMPORTANT: the adapter must travel WITH its trained readout head.
        # Loading an adapter while leaving the stock readout in place silently
        # changes the decision layer underneath the LoRA-trained one.
        assert args.readout, "--adapter requires --readout (they train together)"
        from peft import PeftModel
        d.backbone = PeftModel.from_pretrained(d.backbone, args.adapter)
        d.backbone = d.backbone.to(d.device).eval()
        from safetensors.torch import load_file
        d.readout.load_state_dict(load_file(args.readout))
        print(f"adapter active: {args.adapter} (+ readout {args.readout})", flush=True)

    reqs = [r for _, r in rows]
    t0 = time.time()
    resps = []
    CH = 64
    for i in range(0, len(reqs), CH):
        resps.extend(d.decide(reqs[i:i + CH]))
    print(f"decided {len(reqs)} in {time.time()-t0:.0f}s", flush=True)

    correct = 0
    conf_by, tot_by, confusions = Counter(), Counter(), Counter()
    out = []
    for (gold, _), resp in zip(rows, resps):
        probs = resp["answers"]["q1"]["probabilities"]
        pred = max(probs, key=probs.get)
        ok = int(pred == gold)
        correct += ok
        tot_by[gold] += 1
        conf_by[gold] += probs.get(gold, 0.0)
        if not ok:
            confusions[(gold, pred)] += 1
        out.append({"gold": gold, "pred": pred,
                    "conf": round(probs.get(gold, 0.0), 4), "ok": ok})

    n = len(rows)
    k = len(crit)
    print(f"OOD ACCURACY: {correct}/{n} = {correct/n:.3f}  (random floor = 1/{k} = {1/k:.3f})")
    for t in sorted(tot_by):
        print(f"  {t:<10} acc={sum(1 for r in out if r['gold']==t and r['ok'])}/{tot_by[t]}"
              f" avg_gold_conf={conf_by[t]/tot_by[t]:.3f}")
    if confusions:
        print("top confusions:", confusions.most_common(5))
    else:
        print("top confusions: (none)")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
