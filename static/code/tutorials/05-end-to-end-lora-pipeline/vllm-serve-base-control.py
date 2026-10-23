"""Control leg: serve the BASE dense model via vLLM.

Identical sampling params to vllm-serve-merged.py. Establishes what the vLLM
renderer does WITHOUT any LoRA, so "different from HF" can be attributed to
chat-template rendering rather than silently blaming (or crediting) the adapter.
"""
import json
import os

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

BASE = "Qwen/Qwen3.5-4B"   # or a local snapshot path
RESULTS = "vllm-base-results.json"
PROMPTS = [
    "What is the capital of Australia?",
    'Translate to English: "Bonjour, comment allez-vous?"',
    "Rewrite as a haiku: the ocean at sunrise",
    'Convert to the opposite: "I love this product"',
    "Summarize in one sentence: The company reported strong ",
]


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=BASE,
        quantization="fp8",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        enforce_eager=False,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)
    outs = llm.generate(PROMPTS, sp)
    res = [{"prompt": p, "out": o.outputs[0].text} for p, o in zip(PROMPTS, outs)]
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    for r in res:
        print("PROMPT:", r["prompt"])
        print("  vLLM base :", repr(r["out"][:120]))
    print("DONE_VLLM_BASE")


if __name__ == "__main__":
    main()
