"""Frozen OPS-ROUTE test set against vLLM's native /v1/systemone decision endpoint.

Serves amos-1-small (or any READ_STRATEGIES-whitelisted decision model) with:
  vllm serve <model-dir> --served-model-name amos --port 8127 \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --logprobs-mode raw_logprobs

The benchmark file is the tutorial #6 format: {"teams": {...}, "train": [...], "test": [...]}
with test rows {"text": ..., "team": ...}. Chance floor = 1/len(teams).
"""
import json
import statistics
import urllib.request

URL = "http://127.0.0.1:8127/v1/systemone"
MODEL = "amos"
DATA = "/path/to/opsroute2.json"  # your frozen benchmark, tutorial #6 format

d = json.load(open(DATA))
crit = dict(d["teams"])
rows = d["test"]
ok = 0
masses = []
confs = []
gold_probs = []
for row in rows:
    body = {
        "model": MODEL,
        "state": row["text"],
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Route this on-call incident to the owning team.",
                "criteria": crit,
            }
        },
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    a = resp["answers"]["q1"]
    ok += int(a["choice"] == row["team"])
    confs.append(a["confidence"])
    gold_probs.append(a["probabilities"][row["team"]])
    masses.append(resp["diagnostics"]["q1"]["label_mass"])

n = len(rows)
print(f"ACCURACY: {ok}/{n} = {ok / n:.4f}  (chance floor = {1 / len(crit):.4f})")
print(f"label_mass median={statistics.median(masses):.4f}")
print(f"confidence median={statistics.median(confs):.4f}")
print(f"P(gold) median={statistics.median(gold_probs):.4f}")
