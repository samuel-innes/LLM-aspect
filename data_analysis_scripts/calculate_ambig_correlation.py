"""
Calculate correlation metrics between ambiguity in annotated data / Llama ambiguity model output and BERT entropy
"""

import json
import csv
import re
from sklearn.model_selection import train_test_split
from scipy import stats

def prompt_to_csv(prompt_data_path, csv_out_path):
    # Func from labelled_prompt_to_csv.py
    # load the prompt data
    with open(prompt_data_path, 'r') as fin:
        data = json.load(fin)

    csv_data = []
    dp_id = 0

    pattern = re.compile(r'\".+\"')
    print("len data", len(data))
    for dp in data:
        # get verb from instruction
        match = pattern.search(dp['instruction'])
        if match != None:
            verb = match.group().strip("\"")
            dp_id += 1
            csv_data.append((dp_id, dp['input'], verb))
        else:
            raise RuntimeError
    
    with open(csv_out_path, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(csv_data)

if __name__ == '__main__':
    # # preprocess data
    # prompt_data_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled/llama3_ambig_binary_a1/output.json"
    # csv_out_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled/llama3_ambig_binary_a1/output.csv"
    # prompt_to_csv(prompt_data_path, csv_out_path)

    # prompt_data_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled_ambig_binary/llama3_ambig_binary_a1/output.json"
    # csv_out_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled_ambig_binary/llama3_ambig_binary_a1/output.csv"
    # prompt_to_csv(prompt_data_path, csv_out_path)

    # get the outputs from BERT model
    # see run_test_BERT.sh

    # calculate correlation

    # # load entropy data
    eval_entropies = []
#    /home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/ambig_binary_a1/verb_class_train_entropies.csv
    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/no_habitual/upsampled_a1/verb_class_eval_entropies_BNC.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            eval_entropies.append(float(row[2]))

    train_entropies = []
    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/no_habitual/upsampled_a1/verb_class_train_entropies_BNC.csv", 'r') as f:
    #with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_unlabelled_bert/output_entropies_BNC.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_entropies.append(float(row[2]))
    #print(len(eval_entropies))
    print(len(train_entropies))



    # load labelled data
    eval_labels = []
    #with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled_ambig_binary/llama3_ambig_binary_a1/output.json", 'r') as f:
    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/no_habitual/upsampled_a1/verb_class_eval.json", 'r') as f:

        data = json.load(f)
        for dp in data:
            if dp['output'] == "ambiguous":
                eval_labels.append(1)

            elif dp['output'] == "not-ambiguous":
                eval_labels.append(0)

    train_labels = []
    #with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled/llama3_ambig_binary_a1/output.json", 'r') as f:
    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/no_habitual/upsampled_a1/verb_class_train.json", 'r') as f:

        data = json.load(f)
        for dp in data:
            if dp['output'] == "ambiguous":
                train_labels.append(1)

            elif dp['output'] == "not-ambiguous":
                train_labels.append(0)


    # labels = []
    # entropies = []
    # with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled/llama3_ambig_binary_a1/output.json", 'r') as f:
    #     data = json.load(f)
    #     for dp, entropy in zip(data, train_entropies):
    #         label = dp.get('llama_pred')
    #         if label != None:
    #             if label == "ambiguous":
    #                 labels.append(1)
    #                 entropies.append(entropy)                

    #             elif label == "not-ambiguous":
    #                 labels.append(0)
    #                 entropies.append(entropy) 

    entropies = train_entropies + eval_entropies
    #entropies = train_entropies
    labels = train_labels + eval_labels
    #labels = train_labels
    # # calculate correlation
    eval_corr_eff = stats.pointbiserialr(eval_labels, eval_entropies)
    print("eval:", eval_corr_eff)

    train_corr_eff = stats.pointbiserialr(train_labels, train_entropies)
    print("train:", train_corr_eff)

    both_corr_eff = stats.pointbiserialr(labels, entropies)
    print("both:", both_corr_eff)


