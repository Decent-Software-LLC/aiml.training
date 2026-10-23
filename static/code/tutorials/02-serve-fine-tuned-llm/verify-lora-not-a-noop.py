# 5-minute adapter check — run this before believing any "lora" output
p = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)
base = llm.generate([PROMPT], p)[0].outputs[0].logprobs
lora = llm.generate([PROMPT], p, lora_request=lo)[0].outputs[0].logprobs

same = all(
    [t for t in sb.items() if t[1].logprob == max(v.logprob for v in sf.values())]
        == [t for t in sf.items() if t[1].logprob == max(v.logprob for v in sb.values())]
    for sb, sf in zip(base, lora)
)
if same:
    print("⚠ ADAPTER IS A NO-OP — you're serving the base model. Do not ship this.")
else:
    print("✓ Adapter is active: outputs differ.")
