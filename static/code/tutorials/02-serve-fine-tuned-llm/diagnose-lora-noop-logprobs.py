p = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)
b = llm.generate([PROMPT], p)[0].outputs[0].logprobs        # base
f = llm.generate([PROMPT], p, lora_request=lo)[0].outputs[0].logprobs  # "lora"

top1_flips = 0
max_delta = 0.0
for bstep, fstep in zip(b, f):
    b_top = max(bstep.items(), key=lambda kv: kv[1].logprob)
    f_top = max(fstep.items(), key=lambda kv: kv[1].logprob)
    if b_top[0] != f_top[0]:
        top1_flips += 1
    # logprob assigned by the "lora" run to the base's top-1 token:
    delta = abs(fstep[b_top[0]].logprob - bstep[b_top[0]].logprob)
    max_delta = max(max_delta, delta)

print(f"top-1 flips: {top1_flips}/64, max logprob delta: {max_delta:.4f}")
