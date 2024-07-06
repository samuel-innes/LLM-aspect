# Code adapted from https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19

# Uncomment to install new features that support latest models like Llama 2
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers.git

# When prompted, paste the HF access token you created earlier.
from huggingface_hub import notebook_login
notebook_login()

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

dataset_name = "samuel-innes/UMR-Aspect"
dataset = load_dataset(dataset_name, split='train')
print(len(dataset))
base_model_name = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1 

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

output_dir = "./checkpoints7b"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=300,
    save_steps=20
)

max_seq_length = 512

def formatting_prompts_func(example, use_answer=True):
    output_texts = []
    for i in range(len(example['instruction'])):
        if use_answer:
            text = f"### Question: {example['instruction'][i]} {example['input'][i]}. Just give the class after \"### Answer: \" with no explanation like the following:\n ### Answer: {example['output'][i]}"
        else:
            text = f"### Question: {example['instruction'][i]} {example['input'][i]}"
     
        output_texts.append(text)
    return output_texts

response_template = "### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

if __name__ == '__main__':
    trainer.train()

import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)