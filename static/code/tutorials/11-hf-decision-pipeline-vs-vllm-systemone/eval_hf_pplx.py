#!/usr/bin/env python3
"""HF decision pipeline eval of pplx-decider on the frozen OPS-ROUTE test set.
Same request schema as tutorial #9's eval (state {} + text in instructions,
criteria = full team dict -> 12.5% floor). The model vLLM gated (501) — does
HF's pipeline('decision') take it?

PIN: transformers 5.23.0 (first line with the decision pipeline).
GATE: if the load report prints MISSING for any decision-head parameter
(e.g. proj.bias), the head is randomly initialized — the run measures noise.
"""
import json
from transformers import pipeline

d = json.load(open("traindata/opsroute2.json"))
teams = d["teams"]
test = d["test"]

decide = pipeline("decision", model="path/to/pplx-decider-flash-v1.2")
print("pipeline:", decide)

crit = {t: teams[t] for t in teams}
correct = 0
confs = []
for row in test:
    out = decide({"state": {}, "questions": {"q1": {
        "type": "choice",
        "instructions": ("Route this on-call incident to the owning team.\n" + row["text"]),
        "criteria": crit}}})
    got = out["q1"]["choice"]
    conf = out["q1"]["confidence"]
    confs.append(conf)
    if got == row["team"]:
        correct += 1
confs.sort()
print(f"RESULT: {correct}/{len(test)} = {100*correct/len(test):.1f}% (floor 12.5%)")
print(f"confidence median: {confs[len(confs)//2]:.3f} min: {confs[0]:.3f} max: {confs[-1]:.3f}")
