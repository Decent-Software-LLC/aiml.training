from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="/path/to/models/qwen35-4b",
    quantization="fp8",              # required — bf16 doesn't fit 12 GB
    enable_lora=True,
    max_lora_rank=16,
    max_loras=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=1024,
    enforce_eager=True,
)
lo = LoRARequest("ft", 1, "/path/to/runs/q35-4b-lora/adapter")
out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=96), lora_request=lo)[0]
