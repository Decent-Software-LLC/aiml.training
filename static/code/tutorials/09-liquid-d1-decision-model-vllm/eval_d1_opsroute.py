"""Tutorial #9 OPS-ROUTE eval — aiml.training (writeup:
https://aiml.training/docs/tutorials/liquid-d1-decision-model-vllm/).

Runs the frozen OPS-ROUTE TEST split (tutorial #6's dataset generator,
gen_opsroute_dataset.py -> opsroute2.json) through a vLLM server running a
whitelisted decision model (LiquidAI/d1-3B) over /v1/systemone, zero-shot.

Printed numbers in the article come from exactly this script:
    D1-3B ZERO-SHOT ACCURACY: 23/120 = 0.1917
    label_mass median=0.9067  confidence median=0.1462

The point is the CONTRAST: at-chance accuracy with honest low confidence.
If label_mass is low instead, you are evaluating a broken readout, not a
model -- see tutorial #7 before trusting anything.

Serve first:
    vllm serve LiquidAI/d1-3B --served-model-name d1 \
        --logprobs-mode raw_logprobs --max-model-len 4096 \
        --gpu-memory-utilization 0.85

Run:
    python eval_d1_opsroute.py --data opsroute2.json [--base http://localhost:8000/v1]
"""
import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="opsroute2.json")
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--out", default="d1-opsroute-results.json")
    args = ap.parse_args()

    d = json.load(open(args.data))
    crit = dict(d["teams"])
    rows = d["test"]

    ok, masses, confs, wrong = 0, [], [], []
    for i, row in enumerate(rows):
        body = {
            "model": "d1",
            "state": {},
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Route this on-call incident to the owning team.\n" + row["text"],
                    "criteria": crit,
                }
            },
        }
        req = urllib.request.Request(
            args.base.rstrip("/") + "/systemone",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.load(r)
        got = resp["answers"]["q1"]["choice"]
        hit = got == row["team"]
        ok += int(hit)
        masses.append(resp["diagnostics"]["q1"]["label_mass"])
        confs.append(resp["answers"]["q1"]["confidence"])
        if not hit:
            wrong.append({"i": i, "gold": row["team"], "got": got})

    n = len(rows)
    m = sorted(masses)
    c = sorted(confs)
    print(f"D1-3B ZERO-SHOT ACCURACY: {ok}/{n} = {ok / n:.4f}")
    print(f"label_mass median={m[n // 2]:.4f} min={m[0]:.4f} max={m[-1]:.4f}")
    print(f"confidence median={c[n // 2]:.4f} mean={sum(confs) / n:.4f}")
    json.dump(
        {"accuracy": ok / n, "n": n, "label_mass_median": m[n // 2],
         "confidence_median": c[n // 2], "wrong": wrong},
        open(args.out, "w"), indent=1,
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
