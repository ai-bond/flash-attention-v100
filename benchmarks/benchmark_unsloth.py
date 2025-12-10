### Direct replace unsloth's FA2 import

import os
os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "force"

import sys, types, importlib.util

def make_package(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    m.__package__ = name
    return m

def make_module(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    m.__package__ = name.rpartition(".")[0]
    return m

from flash_attn_v100 import flash_attn_func

fa = make_package("flash_attn")
fa.flash_attn_func = flash_attn_func
sys.modules["flash_attn"] = fa

fai = make_module("flash_attn.flash_attn_interface")
fai.flash_attn_func = flash_attn_func
fai.flash_attn_varlen_func = flash_attn_func
sys.modules["flash_attn.flash_attn_interface"] = fai

fbp = make_module("flash_attn.bert_padding")
fbp.index_first_axis = fbp.pad_input = fbp.unpad_input = lambda x, *a, **k: x
sys.modules["flash_attn.bert_padding"] = fbp

print("🦥 Volta shim installed")

### Set unsloth params

os.environ["WANDB_PROJECT"] = "Your project name"
os.environ["WANDB_API_KEY"] = "Your api key"

hf_id    = "Your Huggingface id"
hf_token = "Your Huggingface api key"

# Choose HF model for tain
original_model   = "unsloth/llama-2-7b-bnb-4bit"
hf_model_split   = original_model.split("/")
hf_model_name    = hf_model_split[0]
hf_model_name_id = hf_model_split[1]

# Unsloth Model params
max_seq_length = 4096
dtype = None
load_in_4bit = True
load_in_8bit = False
full_finetuning = False

### Unsloth monkey patch

import unsloth.models._utils as _utils

_utils.HAS_FLASH_ATTENTION = True
_utils.HAS_FLASH_ATTENTION_SOFTCAPPING = False
_utils.SUPPORTS_BFLOAT16 = False

import unsloth.models.llama as llama_module
import importlib
importlib.reload(llama_module)

if hasattr(llama_module, 'flash_attn_func'):
    print("🦥 flash_attn_func is now in llama.py")
else:
    llama_module.flash_attn_func = flash_attn_func
    print("🦥  Manually patched llama.flash_attn_func")

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{original_model}",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("🦥 Model loaded Volta FA2 is active")

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                                                                                         # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Add for continual pretraining
    lora_alpha = 16,
    lora_dropout = 0,                                                                               # Supports any, but = 0 is optimized
    bias = "none",                                                                                  # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth",                                                         # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,                                                                             # We support rank stabilized LoRA
    loftq_config = None,                                                                            # And LoftQ
)

### Dataset

from datasets import Dataset, load_dataset

alpaca_prompt = """You are an assistant that answers questions. An instruction is given describing a task, along with an input.

1. Follow the instruction as precisely as possible: in terms of type, requested actions, phrasing, and considering the information from the input.
2. In your response, always use the Russian language exclusively. Respond in a literate manner.
3. In your response, do not use other languages.
4. In your response, rely on knowledge from official sources. Consider Laws, encyclopedic knowledge, textbooks, books, official documents, court decisions, rulings, legislative acts.
5. Do not generate random and/or fitting data. In the answer, do not use generic templates. Identify the main point of the question and provide a concise answer.
6. The instruction might be without an input, in which case simply answer within the given context.
7. Follow the same template as provided in the instruction.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = [
        alpaca_prompt.format(instr, inp, out) + EOS
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    out_texts = []
    for ids in tokenizer(texts, add_special_tokens=False).input_ids:
        if len(ids) > max_seq_length - 1:
            ids = ids[:max_seq_length - 1]
        n = (len(ids) // 16) * 16
        if n >= 15:
            ids = ids[:n] + [tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids(EOS)]
            out_texts.append(tokenizer.decode(ids, skip_special_tokens=False))
    return {"text": out_texts}

ds = (
    load_dataset("ai-bond/ru-alpaca-grandpro", split="train[98%:]")
    .map(formatting_prompts_func, batched=True, remove_columns=["instruction", "input", "output"])
    .filter(lambda x: len(tokenizer.encode(x["text"], add_special_tokens=False)) >= 16)
)

### Trainer params

from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=1,
        # num_train_epochs = 1,         # Set this for 1 full training run.
        max_steps=20,
        learning_rate=2e-4,             # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",               # Use this for WandB etc
    ),
)

### Run train

import torch
import time
from unsloth import unsloth_train

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start_event.record()

trainer_stats = unsloth_train(trainer)

end_event.record()
torch.cuda.synchronize()
gpu_time_ms = start_event.elapsed_time(end_event)
print(f"GPU time: {gpu_time_ms / 1000:.2f} sec")

