"""Grade a routing model THROUGH vLLM's /v1/systemone endpoint
(tutorial #7, aiml.training).

See the full writeup: https://aiml.training/docs/tutorials/serve-decision-model-vllm-systemone/

Evaluates the OPS-ROUTE frozen test set from tutorial #6 against a served
model, one POST per row, mirroring EXACTLY the request shape used during
training (empty state; instruction text + incident inside `instructions`;
criteria = team codenames mapped to their squad descriptions).

This is the train/serve-consistency discipline made executable: score
through the path production uses, not the offline harness you trained with.
Two aggregate signals decide whether the integration is healthy:
  accuracy     vs your offline number (a large gap = something structural)
  label_mass   ~1.0 healthy, ~0.0 structurally broken (e.g. no real lm_head)
               -- flagged per-row below; do not ship through this gate.

Dataset: regenerate with gen_opsroute_dataset.py from tutorial #6
(https://aiml.training/code/tutorials/06-fine-tune-decision-router/gen_opsroute_dataset.py)
then pass its JSON path via --data.

Usage:
  python eval_through_endpoint.py --data opsroute2.json \
      [--base-url http://127.0.0.1:8123] [--model router]
"""
import argparse
import json
import urllib.request


def call(base_url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="opsroute2.json path")
    ap.add_argument("--base-url", default="http://127.0.0.1:8123")
    ap.add_argument("--model", default="router")
    args = ap.parse_args()

    d = json.load(open(args.data))
    crit = dict(d["teams"])          # {team_codename: squad description}
    rows = d["test"]                 # FROZEN split; never train on it

    ok = 0
    masses = []
    bad_flags = 0
    for i, row in enumerate(rows):
        body = {
            "model": args.model,
            "state": {},             # match training: everything in q1.instructions
            "questions": {"q1": {
                "type": "choice",
                "instructions": ("Route this on-call incident to the owning "
                                 "team.\n" + row["text"]),
                "criteria": crit,
            }},
        }
        resp = call(args.base_url, body)
        got = resp["answers"]["q1"]["choice"]
        diag = resp["diagnostics"]["q1"]
        hit = got == row["team"]
        ok += int(hit)
        masses.append(diag["label_mass"])
        if not (diag["argmax_is_label"] and diag["label_mass"] > 0.5):
            bad_flags += 1
            print(f"row {i}: !! label_mass={diag['label_mass']:.4f} "
                  f"argmax_is_label={diag['argmax_is_label']} "
                  f"(got={got} gold={row['team']})")

    m = sorted(masses)
    n = len(rows)
    print(f"\nENDPOINT ACCURACY: {ok}/{n} = {ok / n:.4f}")
    print(f"label_mass median={m[n // 2]:.4f} min={m[0]:.4f} max={m[-1]:.4f}")
    if ok / n < 0.5 or m[n // 2] < 0.5:
        print("VERDICT: integration unhealthy. Check, in order: architecture "
              "eligibility vs READ_STRATEGIES, --logprobs-mode raw_logprobs, "
              "lm_head presence in the safetensors index, LoRA attachment "
              "count at merge time. Cheapest detectors first.")
    else:
        print("VERDICT: diagnostics healthy; compare accuracy against the "
              "offline figure before trusting parity.")


if __name__ == "__main__":
    main()
