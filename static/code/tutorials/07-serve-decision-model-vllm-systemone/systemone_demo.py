"""Tutorial #7 demo client for vLLM's native decision endpoint (aiml.training).

See the full writeup: https://aiml.training/docs/tutorials/serve-decision-model-vllm-systemone/

POSTs state + typed questions (choice | noul | score) to /v1/systemone and
prints answers, probabilities, usage, and the diagnostics you must monitor:
  label_mass       ~1.0 means the model is actually deciding among your labels;
                   ~0.0 means the readout is structurally broken (e.g. a model
                   without a real lm_head scored via a fabricated tied head).
  argmax_is_label  False on every row = same failure, different sign.

Serve a gating-eligible checkpoint first (see article, "The gate"), e.g.:
  export VLLM_USE_FLASHINFER_SAMPLER=0
  vllm serve Qwen/Qwen3-4B --served-model-name q35 \
      --logprobs-mode raw_logprobs --max-model-len 2048

Usage: python systemone_demo.py [--base-url http://127.0.0.1:8123] [--model q35]
"""
import argparse
import json
import urllib.request


def decide(base_url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


DEMO = {
    "state": ("Ticket #4412: 'I was charged twice for my subscription this "
              "month and the second charge went to my old card that I "
              "cancelled.'"),
    "instructions": "You route support tickets to the correct team.",
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
            "instructions": "Does this customer need money moved back before "
                            "their next statement?",
            "criteria": {"true": "a refund or reversal is time-sensitive",
                         "false": "no time pressure on funds"},
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this issue?",
            "criteria": ["cosmetic", "minor", "moderate", "severe"],
        },
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8123")
    ap.add_argument("--model", default="q35")
    args = ap.parse_args()

    resp = decide(args.base_url, {"model": args.model, **DEMO})

    print(json.dumps(resp, indent=2))
    print("\n--- tripwires ---")
    ok = True
    for qid, diag in resp["diagnostics"].items():
        flag = "OK " if diag["label_mass"] > 0.5 and diag["argmax_is_label"] else "!! "
        ok &= diag["label_mass"] > 0.5 and diag["argmax_is_label"]
        print(f"{flag}{qid}: label_mass={diag['label_mass']:.4f} "
              f"argmax_is_label={diag['argmax_is_label']}")
    u = resp["usage"]
    print(f"tokens: in={u['input_tokens']} out={u['output_tokens']} "
          f"(one generated token per question)")
    if not ok:
        raise SystemExit("label_mass too low: the served checkpoint is not "
                         "deciding over your labels — check architecture "
                         "eligibility and lm_head presence (see article).")


if __name__ == "__main__":
    main()
