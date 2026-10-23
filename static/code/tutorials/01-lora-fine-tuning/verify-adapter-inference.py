import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

base_path    = "Qwen/Qwen3.5-4B"
adapter_path = "qwen3.5-4b-lora/final_adapter"

# Load the same text-only head we trained with. AutoModelForImageTextToText
# would also resolve this checkpoint — to the multimodal wrapper — but for a
# text-only fine-tune Qwen3_5ForCausalLM keeps the vision tower out of VRAM.
base = Qwen3_5ForCausalLM.from_pretrained(
    base_path,
    load_in_4bit=True,
    device_map="auto",
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

prompt = (
    "Explain in one sentence why we load the text-only head "
    "instead of the multimodal wrapper."
)
inputs = tokenizer.apply_chat_template(
    prompt,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
