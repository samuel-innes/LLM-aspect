from huggingface_hub import notebook_login
notebook_login()

from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#from train2 import formatting_prompts_func
import torch
import re
import os
import json



def get_predictions(dataset, tokenizer, model, device, num_to_class, no_labels=False):
    pred, true = [], []
    num_skipped = 0
    new_dps = []
    for dp in tqdm(dataset):
        new_dp = dp             # carry across all fields from dataset
        try:
            if not no_labels:
                class_to_num[dp['output']]      # sometimes the class in the eval data is incorrect
        except:
            continue
        #text = f"### Question: {dp['instruction']} {dp['input']}"

        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"

        # Combine a prompt with the static strings
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{dp['instruction']}"
        input_context = f"{INPUT_KEY}\n{dp['input']}" if dp["input"] else None
        parts = [part for part in [blurb, instruction, input_context] if part]
        #parts = [part for part in [blurb, input_context] if part]

        # Join prompt template elements into a single string to create the prompt template
        formatted_prompt = "\n\n".join(parts)

        # Store the formatted prompt template in a new key "text"

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #pattern = r"### Solution:\s*([a-zA-Z]+)"
        pattern = r'### Response:\n([a-zA-Z\-]*)'
        matches = re.search(pattern, output_text)
        new_dp["llama_output"] = output_text
        
        if matches:
            pred_label = matches.group(1)
        else:
            pattern = r"### Answer:\s*([a-zA-Z\-]+)"
            matches = re.search(pattern, output_text)
            if matches:
                pred_label = matches.group(1)
            else:
                num_skipped += 1
                new_dps.append(new_dp)
                print("output text", output_text)
                continue
        try:
            new_dp["llama_pred"] = pred_label
            pred.append(class_to_num[pred_label])
            if not no_labels:
                new_dp["comparison"] = pred_label + " - " + dp['output']
                true.append(class_to_num[dp['output']])
            new_dps.append(new_dp)
            
            
        except:
            num_skipped += 1
            print("output text 2", output_text)
            continue
    
    print("num skipped: ", num_skipped)
    if not no_labels:
        print("accuracy", accuracy_score(true, pred))
        print("f1: ", f1_score(true, pred, average='macro'))
        print(classification_report(true, pred, target_names=num_to_class))
    return new_dps, true, pred

if __name__ == '__main__':
    chkpt = 1000

    #model_dir = f"/home/students/innes/ba2/train3_short_results_upsampling_no_inst/checkpoint-{chkpt}"
    #model_dir = f"/home/students/innes/ba2/models/ambig_binary_upsampled_a1/checkpoint-{chkpt}" # if not yet merged
    model_dir = f"/home/students/innes/ba2/models/llama3_normal_desc/final_merged_{chkpt}" # if already merged
    #model_dir = f"/home/students/innes/ba2/models/llama2_with_upsampling/final_merged_{chkpt}" # if already merged

    merge = False
    save_outputs = True
    
    if merge or save_outputs:
        #output_dir = "/home/students/innes/ba2/models/ambig_binary_upsampled_a1/"
        output_dir = "/home/students/innes/ba2/LLM-aspect/data/BNC"
        #output_dir = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled"
    
    no_labels = True
    device_map = {"": 0}
    #base_model_name = "meta-llama/Llama-2-7b-hf"
    base_model_name = "meta-llama/Meta-Llama-3-8B"

    #base_model_name = "meta-llama/Llama-2-7b-chat-hf"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4}
    num_to_class = ["state", "habitual", "activity", "endeavor", "performance"]
    # class_to_num = {"ambiguous":0, "not-ambiguous":1}
    # num_to_class = ["ambiguous", "not-ambiguous"]

    if not merge:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map, torch_dtype=torch.bfloat16, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    #text = "### Question: A mild 2.6 - magnitude earthquake which struck before the landslide may also have helped set off the wall of mud that crashed down on the village , said Rene Solidum , head of the government vulcanology office. instruction: State value corresponds to stative events: no change occurs during the event. The Habitual value is annotated on events that occur regularly in the past or present. The Activity value indicates an event has not necessarily ended and may be ongoing at Document Creation Time (DCT). Endeavor is used for processes that end without reaching completion (i.e., termination), whereas Performance is used for processes that reach a completed result state. Event nominals are annotated as Process. Which class does \"helped\" belong to in this sentence: state, habitual, activity, endeavor, or performance?"

    #dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/umc_unlabelled_ambig_binary/data.json"
    #dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_unlabelled/data.json"
    #dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/upsampled_a1/verb_class_eval.json"
    #dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_unlabelled/data.json"
    #dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/upsampling/verb_class_eval.json"
    dataset_path = "/home/students/innes/ba2/LLM-aspect/data/BNC/prompts.json"
    with open(dataset_path, 'r') as fin:
        dataset = json.load(fin)
    dataset = dataset[:5000]

    # merge Model with Adapter
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

    new_dps, true, pred = get_predictions(dataset, tokenizer, model, device, num_to_class, no_labels)

    if save_outputs:
        with open(os.path.join(output_dir, "BNC_labelled.json"), 'w') as fout:
            json.dump(new_dps, fout)

        print("saved to ", os.path.join(output_dir, "BNC_labelled.json"))

    if not no_labels:
        cm = confusion_matrix(true, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=num_to_class)
        disp.plot()

        plt.savefig("confusion_matrix_llama3.png", dpi=300)