"""vLLM serving of Qwen3.5-4B — base model, FP8."""
import time

if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/path/to/models/qwen35-4b",
        quantization="fp8",          # Marlin FP8 — required to fit 12 GB
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enforce_eager=True,          # no CUDA graph build; simpler, fine for testing
    )

    PROMPT = (
        "You are a customer support agent for a fictional streaming service, "
        "NovaPlay. Respond concisely.\n\n"
        "User: Hi, I was charged twice for my monthly plan. What should I do?\nAssistant:"
    )
    t0 = time.time()
    out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=96))[0]
    dt = time.time() - t0

    n = len(out.outputs[0].token_ids)
    print(out.outputs[0].text)
    print(f"{n} tokens in {dt:.2f}s -> {n/dt:.1f} out tok/s")
