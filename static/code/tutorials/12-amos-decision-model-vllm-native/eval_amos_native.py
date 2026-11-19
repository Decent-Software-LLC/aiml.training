"""Same frozen OPS-ROUTE eval against the Amos authors' own server (amos.serve).

Serve with their pinned env (python 3.12, torch 2.14, transformers 5.17):
  pip install ./amos-1-small/serve
  python -m amos.serve amos-1-small --device cuda --port 8128 --strict --no-state-cache

Diagnostics differ across servers (their server exposes no label_mass; confidence
is computed differently) — compare argmax and accuracy only, not diagnostics.
"""
import json
import statistics
import urllib.request

PORT = "8128"
DATA = "/path/to/opsroute2.json"  # your frozen benchmark, tutorial #6 format

d = json.load(open(DATA))
crit = dict(d["teams"])
rows = d["test"]
ok = 0
confs = []
gold_probs = []
for row in rows:
    body = {
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
        f"http://127.0.0.1:{PORT}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    a = resp["answers"]["q1"]
    ok += int(a["choice"] == row["team"])
    confs.append(a.get("confidence", 0))
    gold_probs.append(a["probabilities"][row["team"]])

n = len(rows)
print(f"ACCURACY: {ok}/{n} = {ok / n:.4f}  (chance floor = {1 / len(crit):.4f})")
print(f"confidence median={statistics.median(confs):.4f}")
print(f"P(gold) median={statistics.median(gold_probs):.4f}")
