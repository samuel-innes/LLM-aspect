"""
This script will run inference on the fine-tuned Llama 2 model on the English UMC data.
"""

from huggingface_hub import notebook_login
notebook_login()

from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
#from train2 import formatting_prompts_func
import torch
import re
import os
import json

from inference import get_predictions

if __name__ == '__main__':
    chkpt = 1000

    #model_dir = f"/home/students/innes/ba2/train3_short_results_upsampling_no_inst/checkpoint-{chkpt}"
    #model_dir = f"/home/students/innes/ba2/models/llama3_ambig_binary_a1/checkpoint-{chkpt}"
    model_dir = f"/home/students/innes/ba2/models/llama3_ambig_binary_a1/final_merged_{chkpt}" # if already merged

    merge = False
    save_outputs = True

    output_dir = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled/llama3_ambig_binary_a1"
    device_map = {"": 0}
    #base_model_name = "meta-llama/Llama-2-7b-hf"
    #base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    base_model_name = "meta-llama/Meta-Llama-3-8B"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4}
    # num_to_class = ["state", "habitual", "activity", "endeavor", "performance"]
    class_to_num = {"ambiguous":0, "not-ambiguous":1}
    num_to_class = ["ambiguous", "not-ambiguous"]

    if not merge:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map, torch_dtype=torch.bfloat16, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    #text = "### Question: A mild 2.6 - magnitude earthquake which struck before the landslide may also have helped set off the wall of mud that crashed down on the village , said Rene Solidum , head of the government vulcanology office. instruction: State value corresponds to stative events: no change occurs during the event. The Habitual value is annotated on events that occur regularly in the past or present. The Activity value indicates an event has not necessarily ended and may be ongoing at Document Creation Time (DCT). Endeavor is used for processes that end without reaching completion (i.e., termination), whereas Performance is used for processes that reach a completed result state. Event nominals are annotated as Process. Which class does \"helped\" belong to in this sentence: state, habitual, activity, endeavor, or performance?"

    # dataset_name = "samuel-innes/UMR-aspect-upsampled"
    # dataset = load_dataset(dataset_name, split='test')

    dataset_name = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_unlabelled/data.json"
    dataset = load_dataset('json', data_files=dataset_name)
    dataset = dataset['train']

    #dataset = dataset[:10]


    # Merge Model with Adapter
    if merge:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model.config.use_cache = False
        model = PeftModel.from_pretrained(
            model, # The base model with full precision
            os.path.join(output_dir, f"checkpoint-{chkpt}"), # Path to the finetuned adapter
        )

        model = model.merge_and_unload()
        model.save_pretrained(os.path.join(output_dir, f"final_merged_{chkpt}"), safe_serialization=False)

    new_dps, true, pred = get_predictions(dataset, tokenizer, model, device, num_to_class, no_labels=True)

    if save_outputs:
        with open(os.path.join(output_dir, "output.json"), 'w') as fout:
            json.dump(new_dps, fout)

        print("saved")