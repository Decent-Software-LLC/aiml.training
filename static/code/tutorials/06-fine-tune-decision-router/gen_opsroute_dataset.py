"""OPS-ROUTE dataset generator — tutorial #6 (aiml.training).

Convention-shaped routing that no pretrained model can guess:
teams are opaque codenames ("squad K"), gold comes from a hidden service x
region chart plus priority keyword overrides. Knowledge is learnable ONLY
from these labelled examples -- which is exactly what fine-tuning is for.

Run:  python gen_opsroute_dataset.py     -> opsroute2.json next to the script
"""
import json
import random

TEAMS = {
    "kestrel":   "squad K",
    "onyx":      "squad O",
    "ember":     "squad E",
    "basalt":    "squad B",
    "verdigris": "squad V",
    "cinder":    "squad C",
    "harrier":   "squad H",
    "marlin":    "squad M",
}

# ---- HIDDEN CONVENTIONS (never shown to any model; only encoded in labels) ----
SERVICES = ["gimbal", "sprocket", "thicket", "lantern", "cobbler"]
REGIONS = ["us-atl", "eu-fra", "ap-sin", "sa-gru"]

# service x region -> owning squad: an arbitrary house chart, NOT derivable
# from any of the names involved. This is the tribal knowledge being taught.
CHART = {
    "gimbal":   {"us-atl": "kestrel",  "eu-fra": "onyx",     "ap-sin": "ember",   "sa-gru": "basalt"},
    "sprocket": {"us-atl": "verdigris","eu-fra": "cinder",   "ap-sin": "harrier", "sa-gru": "marlin"},
    "thicket":  {"us-atl": "onyx",     "eu-fra": "harrier",  "ap-sin": "kestrel", "sa-gru": "cinder"},
    "lantern":  {"us-atl": "basalt",   "eu-fra": "marlin",   "ap-sin": "cinder",  "sa-gru": "kestrel"},
    "cobbler":  {"us-atl": "harrier",  "eu-fra": "ember",    "ap-sin": "basalt",  "sa-gru": "verdigris"},
}

# Keyword overrides: an incident mentioning these tokens goes to a fixed squad,
# overriding the chart -- and each other, by priority order.
KEYWORD_OVERRIDES = [
    ("audit",       "verdigris"),   # highest priority
    ("quarantine",  "onyx"),
    ("runbook",     "marlin"),
]
OVERRIDE_TOKENS = [k for k, _ in KEYWORD_OVERRIDES]

# TRAIN and TEST brand vocabularies are DISJOINT: a test brand is never seen
# during training, so a high test score proves convention LEARNING, not
# sentence memorization.
BRANDS_TRAIN = ["ZephyrPay", "Quantly", "Bramblebox", "Orbitrow", "Fernpost"]
BRANDS_TEST = ["Cloudberry", "Tessera", "Mothlight"]

# EVERY template must surface BOTH {s} and {r}. The gold label is
# CHART[service][region], so any template that hides the region makes its
# label unlearnable-in-principle and silently caps achievable accuracy at
# ~0.45 no matter how well you train. (This exact bug cost us two full
# training runs -- see the article.)
SYMPTOMS = [
    "error rate spiked to 12 percent on the {s} endpoints in {r}",
    "{s} in {r}: latency p99 crossed 8 seconds after last night deploy",
    "{s} health checks are flapping between healthy and down in {r}",
    "queued jobs for {s} in {r} backing up faster than workers drain them",
    "partial outage reported by tenants using {s} in {r}",
    "{s} dashboards show elevated 500 responses in {r}",
]
TAILS = ["", " Paged at 04:12 UTC.", " Multiple tenants escalating.",
         " Severity SEV-2.", " Started after config push.", " One region only."]


def route(text, svc, region):
    """The hidden ground truth: overrides first (priority order), then chart."""
    low = text.lower()
    for tok, squad in KEYWORD_OVERRIDES:
        if tok in low:
            return squad
    return CHART[svc][region]


def make(split, n, seed):
    rnd = random.Random(seed)
    brands = BRANDS_TRAIN if split == "train" else BRANDS_TEST
    items = []
    while len(items) < n:
        svc = rnd.choice(SERVICES)
        reg = rnd.choice(REGIONS)
        text = rnd.choice(SYMPTOMS).format(s=svc, r=reg) + rnd.choice(TAILS)
        text = f"[{rnd.choice(brands)}] {text}"
        # ~18% of rows carry a keyword token to teach the override rules
        if rnd.random() < 0.18:
            tok = rnd.choice(OVERRIDE_TOKENS)
            text += rnd.choice([
                f" A compliance {tok} flag was attached.",
                f" Ticket tagged {tok.upper()} by triage bot.",
                f" Mentioned on-call note: {tok}.",
            ])
        items.append({"text": text, "team": route(text, svc, reg)})
    rnd.shuffle(items)
    return items


if __name__ == "__main__":
    train, test = make("train", 800, 7), make("test", 120, 99)
    out = {"teams": TEAMS, "train": train, "test": test}
    with open("opsroute2.json", "w") as f:
        json.dump(out, f)
    from collections import Counter
    print("train dist:", dict(Counter(r["team"] for r in train)))
    print("test  dist:", dict(Counter(r["team"] for r in test)))
    print("override rows in train:",
          sum(1 for r in train if any(t in r["text"].lower() for t in OVERRIDE_TOKENS)))
