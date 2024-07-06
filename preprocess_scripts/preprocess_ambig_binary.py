"""
Create prompts for Llama model from ambiguity prediction data with upsamping
"""

import csv
import pickle 
import json
from preprocess_data import dp_to_prompt, load_umr_data
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # load the csv data from the file
    with open("/home/students/innes/ba2/LLM-aspect/data/upsampling_data/ambig_upsampling.csv") as f:
        reader = csv.reader(f)
        line_count = 0
        upsampling_prompts = []
        for line in reader:
            if line_count == 0:
                print("Columns are: ", line)
                line_count += 1

            else:
                sentence = line[0]
                verb = line[1]
                dp = {}
                # instruction
                definitions = "Verbal aspect in language indicates how an action unfolds over time, emphasizing its internal structure, such as whether it is ongoing, completed, repeated, or momentary, distinct from tense which specifies when the action occurs relative to a reference point. Some sentences have multiple possible readings leading to different aspect interpretations."    
                question = f"Does \"{verb}\" have an ambiguous aspect reading in this sentence (i.e. multiple possible aspect classes)?"
                dp["instruction"] = definitions + " " + question
            
                # input
                dp['input'] = sentence
                dp['output'] = "ambiguous"
                upsampling_prompts.append(dp)
                line_count += 1



    # add that to the other data
    pkl_out_path="/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/upsampling/data.pkl"
    with open(pkl_out_path, 'rb') as fin:
        non_upsampling_data = pickle.load(fin)



    prompts = dp_to_prompt(non_upsampling_data, no_labels = False, with_ambiguous_class = True, annotations_present=True, with_habitual=False)
    print("len prompts without upsampling:", len(prompts))
    prompts += upsampling_prompts
    print("len upsampling :", len(upsampling_prompts))

    # save to data/preprocessed/ambiguity_classification/umr/upsampled_a1
    prompt_out_train_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/upsampled_a1/verb_class_train.json"
    prompt_out_eval_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/upsampled_a1/verb_class_eval.json"
    
    # train_data, eval_data = train_test_split(prompts, test_size=0.3, random_state=1, shuffle=True)
    
    # with open(prompt_out_train_path, 'w') as fout_train:
    #     json.dump(train_data, fout_train)
    # with open(prompt_out_eval_path, 'w') as fout_eval:
    #     json.dump(eval_data, fout_eval)
    
    # print("Prompt train files written to:", prompt_out_train_path)
    # print("Prompt eval files written to:", prompt_out_eval_path)