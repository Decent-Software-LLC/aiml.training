"""Tutorial #9 d1 demo — aiml.training (writeup:
https://aiml.training/docs/tutorials/liquid-d1-decision-model-vllm/).

Posts one support-ticket state with three typed questions (choice / noul /
score) to a vLLM server running LiquidAI/d1-3B and prints the answers plus
the calibration diagnostics. Read the diagnostics FIRST: label_mass near 1
and argmax_is_label true before trusting any probability.

Serve first (12 GB card):
    vllm serve LiquidAI/d1-3B --served-model-name d1 \
        --logprobs-mode raw_logprobs --max-model-len 4096 \
        --gpu-memory-utilization 0.85

Run:
    python d1_systemone_demo.py [--base http://localhost:8000/v1]
"""
import argparse
import json
import urllib.request

REQ = {
    "model": "d1",
    "state": (
        "Ticket #4412: 'I was charged twice for my subscription this month "
        "and the second charge went to my old card that I cancelled.'"
    ),
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "charges, refunds, invoices, payment methods",
                "tech": "bugs, errors, broken features",
                "account": "logins, profile settings, cancellations",
                "sales": "upgrades, plans, enterprise inquiries",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does this customer need money moved back before their next statement?",
            "criteria": {
                "true": "a refund or reversal is time-sensitive",
                "false": "no time pressure on funds",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this issue?",
            "criteria": ["cosmetic", "minor", "moderate", "severe"],
        },
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    args = ap.parse_args()

    req = urllib.request.Request(
        args.base.rstrip("/") + "/systemone",
        data=json.dumps(REQ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)

    # Gate 1: diagnostics honesty. If label_mass is low or the argmax is not a
    # label, the probabilities below are noise -- stop and fix the model/endpoint.
    diag = resp.get("diagnostics", {})
    for q, d in diag.items():
        lm, ok = d.get("label_mass", 0.0), d.get("argmax_is_label", False)
        flag = "OK" if (lm > 0.5 and ok) else "UNTRUSTWORTHY READOUT"
        print(f"[diag] {q}: label_mass={lm:.4f} argmax_is_label={ok} -> {flag}")

    print(json.dumps(resp["answers"], indent=2))
    print("usage:", resp.get("usage"))


if __name__ == "__main__":
    main()
