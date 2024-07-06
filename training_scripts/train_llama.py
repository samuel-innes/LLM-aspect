# source https://medium.com/@kshitiz.sahay26/fine-tuning-llama-2-for-news-category-prediction-a-step-by-step-comprehensive-guide-to-deeccf3e3a88

import os
import re
from random import randrange
from functools import partial
import torch
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4, "ambiguous":5}
# num_to_class = ["state", "habitual", "activity", "endeavor", "performance", "ambiguous"]

class_to_num = {"ambiguous":0, "not-ambiguous":1}
num_to_class = ["ambiguous", "not-ambiguous"]

def create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    """
    Configures model quantization method using bitsandbytes to speed up training and inference

    :param load_in_4bit: Load model in 4-bit precision mode
    :param bnb_4bit_use_double_quant: Nested quantization for 4-bit model
    :param bnb_4bit_quant_type: Quantization data type for 4-bit model
    :param bnb_4bit_compute_dtype: Computation data type for 4-bit model
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

    return bnb_config

def load_model(model_name, bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """

    # Get number of GPU device and set maximum memory
    #n_gpus = torch.cuda.device_count()
    #max_memory = f'{40960}MB'

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        #device_map = "auto", # dispatch the model efficiently on the available resources
        #max_memory = {i: max_memory for i in range(n_gpus)},
    )

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)

    # Set padding token as EOS token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


################################################################################
# transformers parameters
################################################################################

# The pre-trained model from the Hugging Face Hub to load and fine-tune
#model_name = "meta-llama/Llama-2-7b-hf"
model_name = "meta-llama/Meta-Llama-3-8B"

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = True

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Compute data type for 4-bit base models
bnb_4bit_compute_dtype = torch.bfloat16

# Load model from Hugging Face Hub with model name and bitsandbytes configuration

bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)

model, tokenizer = load_model(model_name, bnb_config)
# dataset_name = "samuel-innes/UMR-aspect-upsampled"
# train_dataset = load_dataset(dataset_name, split='train')
# eval_dataset = load_dataset(dataset_name, split='test')

train_dataset = load_dataset('json', data_files="/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/upsampled_a1/verb_class_train.json")
eval_dataset = load_dataset('json', data_files="/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/upsampled_a1/verb_class_eval.json")

train_dataset = train_dataset['train']
eval_dataset = eval_dataset['train']

def create_prompt_formats(sample):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """

    # Initialize static strings for the prompt template
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{INTRO_BLURB}"
    #instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    #parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    parts = [part for part in [blurb, input_context, response, end] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample

create_prompt_formats(train_dataset[randrange(len(train_dataset))])
create_prompt_formats(eval_dataset[randrange(len(eval_dataset))])

def get_max_length(model):
    """
    Extracts maximum token length from the model configuration

    :param model: Hugging Face model
    """

    # Pull model configuration
    conf = model.config
    # Initialize a "max_length" variable to store maximum sequence length as null
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = ["instruction", "input", "output", "text"],
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed = seed)

    return dataset

# Random seed
seed = 42

max_length = get_max_length(model)
preprocessed_train_dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset)
preprocessed_eval_dataset = preprocess_dataset(tokenizer, max_length, seed, eval_dataset)

def create_peft_config(r, lora_alpha, target_modules, lora_dropout, bias, task_type):
    """
    Creates Parameter-Efficient Fine-Tuning configuration for the model

    :param r: LoRA attention dimension
    :param lora_alpha: Alpha parameter for LoRA scaling
    :param modules: Names of the modules to apply LoRA to
    :param lora_dropout: Dropout Probability for LoRA layers
    :param bias: Specifies if the bias parameters should be trained
    """
    config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )

    return config


def find_all_linear_names(model):
    """
    Find modules to apply LoRA to.

    :param model: PEFT model
    """

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit = False):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )

def print_trainable_parameters(model, use_4bit = False):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )

def fine_tune(model,
          tokenizer,
          train_dataset,
          eval_dataset,
          lora_r,
          lora_alpha,
          lora_dropout,
          bias,
          task_type,
          per_device_train_batch_size,
          gradient_accumulation_steps,
          warmup_steps,
          max_steps,
          learning_rate,
          fp16,
          logging_steps,
          output_dir,
          optim):
    """
    Prepares and fine-tune the pre-trained model.

    :param model: Pre-trained Hugging Face model
    :param tokenizer: Model tokenizer
    :param dataset: Preprocessed training dataset
    """

    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    #model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # Get LoRA module names
    target_modules = find_all_linear_names(model)

    # Create PEFT configuration for these modules and wrap the model to PEFT
    peft_config = create_peft_config(lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        #print("preds1,", preds[0])
        #print("labels1", labels[0])
        pred_list, true_list = [], []
        for pred, label in zip(preds, labels):
            #outputs = model.generate(input_ids=pred["input_ids"].to("cuda"), attention_mask=pred["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
            #output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            label_text = tokenizer.decode(label[label!=-100], skip_special_tokens=True)

            #print("pred", output_text)
            #print(len(preds[0]))
            #print("label_text", label_text)
            try:
                class_to_num['label']
            except:
                continue
            outputs = model.generate(input_ids=pred["input_ids"].to("cuda"), attention_mask=pred["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            label_text = tokenizer.decode(label[0], skip_special_tokens=True)

            print("pred", output_text)
            print("label_text", label_text)
            pattern = r"### Solution:\s*([a-zA-Z]+)"
            pattern = r'### Response:\n([a-z]*)'
            matches = re.search(pattern, output_text)
            
            if matches:
                pred_label = matches.group(1)
            else:
                pattern = r"### Answer:\s*([a-zA-Z]+)"
                matches = re.search(pattern, output_text)
                if matches:
                    pred_label = matches.group(1)
                else:
                    num_skipped += 1
                    print("output text", output_text)
                    continue
            try:
                pred_list.append(class_to_num[pred_label])
                true_list.append(class_to_num[label['output']])
            except:
                num_skipped += 1
                print("output text", output_text)

        # Calculate accuracy
        accuracy = accuracy_score(true_list, pred_list)

        # Calculate precision, recall, and F1-score
        precision = precision_score(true_list, pred_list, average='weighted')
        recall = recall_score(true_list, pred_list, average='weighted')
        f1 = f1_score(true_list, pred_list, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    # Training parameters
    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset=eval_dataset,
        args = TrainingArguments(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = warmup_steps,
            max_steps = max_steps,
            save_steps = 50, 
            #evaluation_strategy="steps",
            #eval_steps=2,
            learning_rate = learning_rate,
            fp16 = fp16,
            logging_steps = logging_steps,
            output_dir = output_dir,
            optim = optim,
        ),
        #compute_metrics=compute_metrics,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
    )

    model.config.use_cache = False

    do_train = True

    # Launch training and log metrics
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # Save model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok = True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 16

# Alpha parameter for LoRA scaling
lora_alpha = 64

# Dropout probability for LoRA layers
lora_dropout = 0.1

# Bias
bias = "none"

# Task type
task_type = "CAUSAL_LM"

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./models/ambig_binary_upsampled_a1"

# Batch size per GPU for training
per_device_train_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Optimizer to use
optim = "paged_adamw_32bit"

# Number of training steps (overrides num_train_epochs)
max_steps = 1000

# Linear warmup steps from 0 to learning_rate
warmup_steps = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True

# Log every X updates steps
logging_steps = 1

fine_tune(model,
      tokenizer,
      preprocessed_train_dataset,
      preprocessed_eval_dataset,
      lora_r,
      lora_alpha,
      lora_dropout,
      bias,
      task_type,
      per_device_train_batch_size,
      gradient_accumulation_steps,
      warmup_steps,
      max_steps,
      learning_rate,
      fp16,
      logging_steps,
      output_dir,
      optim)