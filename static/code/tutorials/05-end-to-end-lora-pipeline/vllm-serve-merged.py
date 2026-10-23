"""Final leg: serve the MERGED DENSE checkpoint via vLLM.

No enable_lora -- the LoRA is already baked into the weights -- which is what
bypasses vLLM #49354 (hybrid-GDN LoRA loaded but silently not applied).
Compares vLLM outputs against the HF base/HF merged outputs recorded by
merge-servable.py (merge-compare.json).
"""
import json
import os

# vLLM 0.32's registry JIT needs CPython headers (skip if your distro ships them)
os.environ.setdefault("CPATH", "/tmp/usr/include:/tmp/usr/include/python3.11")

MERGED = "runs/q35-e2e/merged-dense-bf16"          # from merge-servable.py
COMPARE = "runs/q35-e2e/serve/merge-compare.json"  # written by merge-servable.py


def main():
    from vllm import LLM, SamplingParams
    # vLLM 0.32 + Qwen3.5: flashinfer's sampler JIT-crashes on this box; the
    # native sampler is the supported fallback. Harmless if unneeded.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    llm = LLM(
        model=MERGED,
        quantization="fp8",           # bf16 dense won't fit 12 GB with KV cache
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        enforce_eager=False,
        trust_remote_code=True,
    )
    print("VLLM_LOADED: True", flush=True)
    with open(COMPARE) as f:
        comp = json.load(f)
    hf_base = comp["base"]
    hf_merged = comp["merged"]

    sp = SamplingParams(max_tokens=64, temperature=0.0)
    outputs = llm.generate(list(hf_base.keys()), sp)
    vllm_merged = {}
    for o in outputs:
        vllm_merged[o.prompt] = o.outputs[0].text.strip()

    same_as_hf = 0
    n = len(hf_base)
    for p in hf_base:
        vb = hf_base[p]
        vm = hf_merged[p]
        vv = vllm_merged[p]
        match_hf = vv == vm
        same_as_hf += int(match_hf)
        print(f"\nPROMPT: {p[:55]}")
        print(f"  HF base   : {vb[:90]!r}")
        print(f"  HF merged : {vm[:90]!r}")
        print(f"  vLLM dens : {vv[:90]!r}")
        print(f"  vLLM==HF_merged: {match_hf}   vLLM==HF_base: {vv == vb}")

    print(f"\nVLLM_MATCH_HF_MERGED: {same_as_hf}/{n}")
    print("DONE_VLLM_DENSE", flush=True)


if __name__ == "__main__":
    main()
