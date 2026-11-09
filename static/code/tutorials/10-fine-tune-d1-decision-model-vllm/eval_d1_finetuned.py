"""Tutorial #10 fine-tuned eval — aiml.training (writeup:
https://aiml.training/docs/tutorials/fine-tune-d1-decision-model-vllm/).

Runs the frozen OPS-ROUTE TEST split (tutorial #6's dataset generator, same
seed discipline -- never trained on) against the MERGED d1-3B+LoRA checkpoint
served natively on /v1/systemone. Same script shape as #9's stock eval: same
120 examples, same eight-squad option list (chance floor 12.5%), client-side
label_mass recomputed from raw logprobs.

Measured on one RTX 3080 Ti (vLLM 0.33, --logprobs-mode raw_logprobs):
  D1-3B FINE-TUNED ACCURACY: 119/120 = 0.9917
  label_mass median=0.9999 min=0.9995 max=1.0000
  confidence median=0.9986 mean=0.9859
Stock d1-3B on the same split scored 23/120 (tutorial #9).

Prereq: merged checkpoint served as in the article, e.g.
  vllm serve ./models/d1-3b-opsroute --served-model-name d1-opsroute \
    --max-model-len 4096 --gpu-memory-utilization 0.85 \
    --logprobs-mode raw_logprobs --trust-remote-code
"""
import json
import math
import urllib.request

URL = "http://127.0.0.1:8125/v1/systemone"
MODEL = "d1-opsroute"
DATA = "/home/agent/tierA/traindata/opsroute2.json"  # tutorial #6 generator

d = json.load(open(DATA))
crit = dict(d["teams"])  # all eight squads -> 12.5% chance floor
rows = d["test"]

ok = 0
masses, confs = [], []
for row in rows:
    body = {"model": MODEL, "state": {}, "questions": {"q1": {
        "type": "choice",
        "instructions": ("Route this on-call incident to the owning team.\n"
                         + row["text"]),
        "criteria": crit,
    }}}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    a = r["answers"]["q1"]
    got = a.get("answer", "")
    want = row["team"]
    ok += (got == want)
    # verify vendor diagnostics from raw logprobs (never trust, always recompute)
    lp = a.get("logprobs") or {}
    toks = lp.get("content") or []
    mass = sum(math.exp(t["logprob"]) for t in toks if t.get("label"))
    top = max((math.exp(t["logprob"]) for t in toks), default=0.0)
    masses.append(mass)
    confs.append(top)

masses.sort()
confs.sort()
n = len(masses)
print(f"D1-3B FINE-TUNED ACCURACY: {ok}/{n} = {ok/n:.4f}")
print(f"label_mass median={masses[n//2]:.4f} min={masses[0]:.4f} max={masses[-1]:.4f}")
print(f"confidence median={confs[n//2]:.4f} mean={sum(confs)/n:.4f}")
