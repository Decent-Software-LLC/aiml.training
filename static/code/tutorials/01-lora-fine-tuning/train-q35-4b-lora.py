"""
LoRA fine-tuning of Qwen/Qwen3.5-4B on a single 12 GB GPU (RTX 3080 Ti).

Verified run (2026-10-19), exact stack:
  torch 2.13.0+cu130, transformers 5.19.0, peft 0.21.2,
  bitsandbytes 0.50.2, trl 1.16.0

Result of that run:
  500 Dolly examples x 3 epochs, batch 1 x grad-accum 16
  -> 93 optimizer steps, 279 s wall time, 1367 tok/s
  -> 21.2M trainable params, peak training VRAM 6.70 GB
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3.5-4B"
OUTPUT_DIR   = "qwen3.5-4b-lora"
SEQ_LENGTH   = 512       # the model supports 262k ctx; 512 keeps activations small
BATCH_SIZE   = 1         # per-device
GRAD_ACCUM   = 16        # effective batch = 16
NUM_EPOCHS   = 3
LR           = 2e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
N_TRAIN_EX   = 500       # our verified run used 500 Dolly examples

# ─────────────────────────────────────────────
# 1. TOKENIZER
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Vocab size: {tokenizer.vocab_size}")   # 248k
print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
# Pull the data straight from the hub. Dolly's real columns are
# instruction / context / response / category (not input/output). The same
# split the verified run used: 500 train rows (0..500) plus a disjoint eval
# tail from rows 14000..15000 of the train split.
raw = load_dataset("databricks/databricks-dolly-15k", split="train").select(range(N_TRAIN_EX))
eval_raw = load_dataset("databricks/databricks-dolly-15k", split="train")
eval_raw = eval_raw.select(range(14000, 15000))
print(f"Train size: {len(raw)}, eval size: {len(eval_raw)}")

# trl 1.x's SFTTrainer reads either a `text` column (plain LM), `prompt` +
# `completion` columns, or a `messages` column (chat). Dolly is none of those —
# it's instruction/context/response. We convert to `messages`, a two-turn
# conversation, so the model is trained *through its chat template* — the same
# rendering it will be served with in the next tutorial (train/serve consistency).
def to_messages(ex):
    # Same column names + join as the eval script (instruction/context/response).
    prompt = (ex["instruction"] + (("\n\n" + ex["context"]) if ex.get("context") else "")).strip()
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ex["response"]},
    ]}
raw = raw.map(to_messages)
eval_raw = eval_raw.map(to_messages)

# ─────────────────────────────────────────────
# 3. MODEL (4-bit NF4, TEXT-ONLY HEAD)
# ─────────────────────────────────────────────
# Qwen3.5 gotcha, the big one: the model on the hub is NOT a plain
# transformer. The default entry point is Qwen3_5ForConditionalGeneration —
# a VLM with a 27-layer vision tower sitting on top of the LLM backbone.
# That backbone is itself hybrid: 24 GatedDeltaNet linear-attention layers
# plus 8 full-attention layers (one full-attention layer for every three
# linear-attention layers), with mrope position embeddings and a 262k
# context window.
#
# Our fine-tune is text-only, so we load the text-only head
# Qwen3_5ForCausalLM directly. If you go through the multimodal wrapper
# instead, the entire vision tower parks in VRAM for no reason — several
# GBs you do not have on a 12 GB card.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,    # Qwen3.5 is bf16-native
    bnb_4bit_use_double_quant=True,           # NF4 + double quant keeps the footprint down
)

# No flash kernels on purpose: causal-conv1d and fla fail to build on this
# CUDA build. The reference GatedDeltaNet kernels ship with transformers
# and are sufficient; the full-attention layers use PyTorch SDPA.
print("Loading Qwen3_5ForCausalLM with 4-bit NF4...")
model = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
)

# Disable KV cache during training to avoid memory bloat.
model.config.use_cache = False
# Required for gradient checkpointing + LoRA to work together.
model.enable_input_require_grads()

# ─────────────────────────────────────────────
# 4. LoRA CONFIGURATION
# ─────────────────────────────────────────────
# This is the standard LLaMA-style LoraConfig — nothing Qwen-specific about
# the config object itself, only about which module names exist:
#
#   Full-attention layers (8):   q_proj, k_proj, v_proj, o_proj
#   MLPs (all layers):           gate_proj, up_proj, down_proj
#   GatedDeltaNet linear-attention layers (24):
#                                in_proj_qkv, in_proj_z, in_proj_b,
#                                in_proj_a, out_proj
#
# Note the two "families" of projection names: q/k/v/o_proj exist only in
# the 8 full-attention layers, while in_proj_* / out_proj exist only in the
# 24 linear-attention layers. Listing all of them is what makes the adapter
# cover the whole backbone.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        # full-attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLPs
        "gate_proj", "up_proj", "down_proj",
        # GatedDeltaNet linear-attention layers
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Our run: trainable params: 21.2M

# ─────────────────────────────────────────────
# 5. TRAINING (SFTTrainer)
# ─────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    bf16=True,                     # Qwen3.5 is bf16-native
    gradient_checkpointing=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    report_to="none",
    optim="adamw_torch",
    warmup_steps=3,                   # ~3% warmup of the 93-step run (transformers 5.x: warmup_ratio is gone)
    lr_scheduler_type="cosine",
    max_length=512,                   # trl 1.x: truncation now lives in SFTConfig
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=raw,             # 2-turn `messages` (user: instruction+context, assistant: response)
    eval_dataset=eval_raw,        # rows 14000..15000 of Dolly, disjoint from the train rows
)

trainer.train()
# Our run: 93 steps, 279 s wall time, 1367 tok/s

# ─────────────────────────────────────────────
# 6. SAVE ADAPTER
# ─────────────────────────────────────────────
adapter_path = f"{OUTPUT_DIR}/final_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"Adapter saved to {adapter_path}")
