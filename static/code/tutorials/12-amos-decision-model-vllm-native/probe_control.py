"""Two probes that make a low benchmark score interpretable.

1. CONTROL (garbage options): a sentence-level choice question the readout has no
   reason to fail. If P(cat) is high, the plumbing works and any low benchmark
   score is a knowledge problem, not a serving bug.
2. PREDICTION DISTRIBUTION: counts predicted-vs-gold labels over the frozen test
   set. Skew (one label predicted 3x its gold rate, the top gold label predicted
   once) tells you the model has a vocabulary bias, not a uniform confusion.

Works against vLLM's /v1/systemone (pass "model" in the body) — for the authors'
amos.serve, drop the "model" key and point at its port.
"""
import collections
import json
import urllib.request

URL = "http://127.0.0.1:8127/v1/systemone"
MODEL = "amos"
DATA = "/path/to/opsroute2.json"


def ask(state, instructions, criteria):
    body = {"model": MODEL, "state": state,
            "questions": {"q": {"type": "choice",
                                "instructions": instructions,
                                "criteria": criteria}}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["answers"]["q"]


# 1. Control: semantics, not structure.
a = ask("The cat sat on the mat.",
        "What animal is in the sentence?",
        {"banana": None, "giraffe": None, "cat": None})
print("CONTROL:", a["choice"], {k: round(v, 4) for k, v in a["probabilities"].items()})

# 2. Prediction distribution vs gold distribution.
d = json.load(open(DATA))
crit = dict(d["teams"])
preds = collections.Counter()
gold = collections.Counter()
for row in d["test"]:
    a = ask(row["text"], "Route this on-call incident to the owning team.", crit)
    preds[a["choice"]] += 1
    gold[row["team"]] += 1
print("pred dist:", dict(preds.most_common()))
print("gold dist:", dict(gold.most_common()))
top = gold.most_common(1)[0]
print(f"always-{top[0]} baseline would score: {top[1]}/{sum(gold.values())}")
