from huggingface_hub import notebook_login
notebook_login()

from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os

def create_prompt(sentence, verb, ambig_binary=True):
    dp = {}
    if ambig_binary:
        # instruction
        #definitions = "Verbal aspect in language indicates how an action unfolds over time, emphasizing its internal structure, such as whether it is ongoing, completed, repeated, or momentary, distinct from tense which specifies when the action occurs relative to a reference point. Some sentences have multiple possible readings leading to different aspect interpretations."    
        #definitions = "Verbal aspect in language indicates how an action unfolds over time, emphasizing its internal structure, such as whether it is ongoing, completed, repeated, or momentary, distinct from tense which specifies when the action occurs relative to a reference point. Some sentences have multiple possible readings leading to different aspect interpretations."    
        definitions = "The annotation distinguishes five base level aspectual values — state, habitual, activity, endeavor, and performance. The State value corresponds to stative events: no change occurs during the event. It also includes predicate nominals (be a doctor), predicate locations (be in the forest), and thetic (presentational) possession (have a cat). The Habitual value is annotated on events that occur regularly in the past or present. The Activity value indicates an event has not necessarily ended and may be ongoing at Document Creation Time (DCT). Endeavor is used for processes that end without reaching completion (i.e., termination), whereas Performance is used for processes that reach a completed result state. Event nominals are typically hard to annotate for aspect, since they lack the grammatical cues that verbs often show. Therefore, they are all annotated with the coarse-grained value Process."

        #question = f"Does \"{verb}\" have an ambiguous aspect reading in this sentence (i.e. multiple possible aspect classes)?"
        question = f"Which class does \"{verb}\" belong to in this sentence: state, habitual, activity, endeavor or performance?"

        dp["instruction"] = definitions + " " + question
    
        # input
        dp['input'] = sentence
    else:
        raise NotImplementedError
    return dp



def get_single_sentence_prediction(dp, device, model, tokenizer):
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


    if matches:
        pred_label = matches.group(1)
    else:
        pattern = r"### Answer:\s*([a-zA-Z\-]+)"
        matches = re.search(pattern, output_text)
        if matches:
            pred_label = matches.group(1)
        else:
            print("Model gave output in incorrect format. Model output: ", output_text)
            pred_label = None

    return pred_label

    




if __name__ == '__main__':
    chkpt = 1000

    #model_dir = f"/home/students/innes/ba2/train3_short_results_upsampling_no_inst/checkpoint-{chkpt}"
    #model_dir = f"/home/students/innes/ba2/models/llama2_no_inst/checkpoint-{chkpt}" # if not yet merged
    #model_dir = f"/home/students/innes/ba2/models/llama3_ambig_binary_a1/final_merged_{chkpt}" # if already merged
    model_dir = f"/home/students/innes/ba2/models/llama3_normal_desc/final_merged_{chkpt}"
    merge = False
    
    #output_dir = "/home/students/innes/ba2/models/llama3_ambig_binary_a1/"
    output_dir = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/umc_labelled/llama3_ambig_binary_a1"
    device_map = {"": 0}
    #base_model_name = "meta-llama/Llama-2-7b-hf"
    base_model_name = "meta-llama/Meta-Llama-3-8B"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4}
    num_to_class = ["state", "habitual", "activity", "endeavor", "performance"]
    # class_to_num = {"ambiguous":0, "not-ambiguous":1}
    # num_to_class = ["ambiguous", "not-ambiguous"]

    if not merge:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map, torch_dtype=torch.bfloat16, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

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
    sentence = "He was writing his paper yesterday."
    verb = "writing"
    dp = create_prompt(sentence, verb)
    print(get_single_sentence_prediction(dp, device, model, tokenizer))