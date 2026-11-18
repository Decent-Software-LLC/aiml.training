#!/usr/bin/env python3
"""Decision-model support matrix probe: check BOTH stacks' gates BEFORE
downloading any weights. Run once per stack venv.

vLLM side: model registry + READ_STRATEGIES (the /v1/systemone guest list).
HF side: MODEL_FOR_DECISION_MAPPING (AutoModelForDecision's gate).

Usage: python probe_matrix.py            # prints both tables for the installed stack
"""
import json, sys

# 1) what the checkpoint declares (no download needed)
def declared(repo_id: str) -> None:
    import urllib.request
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    cfg = json.load(urllib.request.urlopen(url, timeout=20))
    print(repo_id, "| arch:", cfg.get("architectures"), "| model_type:", cfg.get("model_type"))

for repo in ["perplexity/pplx-decider-flash-v1.2", "LiquidAI/d1-3B", "LiquidAI/d1-2B"]:
    declared(repo)

# 2) vLLM gates (if this venv has vllm)
try:
    from vllm import ModelRegistry
    reg = set(ModelRegistry.get_supported_archs())
    from vllm.entrypoints.openai.decisions.strategies import READ_STRATEGIES
    for a in ["D1Model", "Lfm3ForCausalLM", "Lfm3VlForConditionalGeneration",
              "Lfm2VlForConditionalGeneration", "Qwen3_5Model"]:
        print(f"vLLM {a}: model={'SUPPORTED' if a in reg else 'missing'} "
              f"decision={'YES' if a in READ_STRATEGIES else 'no'}")
except ImportError:
    print("vLLM not in this venv")

# 3) HF gate (if this venv has transformers >= 5.23)
try:
    from transformers import MODEL_FOR_DECISION_MAPPING
    print("HF decision mapping:", [c.__name__ for c in MODEL_FOR_DECISION_MAPPING])
except ImportError:
    print("transformers decision pipeline not in this venv")
