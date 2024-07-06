"""
Converts Llama-style prompts to a .csv file for BERT inference.
"""

import json
import re
import csv

from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prompt_to_csv(prompt_data_path, csv_out_path):
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
    
    train_data, eval_data = train_test_split(data, test_size=0.3, random_state=1, shuffle=False)
    with open(csv_out_path, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(csv_data)


def en_fr_to_csv(en_fr_data_path, labelled_en_data_path):
    with open(en_fr_data_path) as fin:
        data = json.load(fin)

    with open(labelled_en_data_path) as fin:
        labelled_en_data = json.load(fin)


    # TODO: change 
    # get dictionary with sent_id-verb_id : label
    sent_verb_id_to_label = {}
    for dp in labelled_en_data:
        label = dp.get('llama_pred')
        if label != None:
            key = str(dp['sent_id']) + "-" +  str(dp['verb_id'])
            sent_verb_id_to_label[key] = label

    # iterate over labelled data

    # get the English data point
    en_csv_dps = []
    fr_csv_dps = []

    labelled_data_pos = 0
    dp_id = 0

    print("len labelled en data", len(labelled_en_data))
    # iterate over aligned data
    sent_id = 0
    for sent in tqdm(data):
        en_sent = sent['sentence']
        fr_sent = sent['fr_sentence']
        unlabelled_sent_id = sent['id']

        verb_id = 0
        for verb in sent['verbs']:
            en_verb = verb[0]
            fr_verb = verb[1]
            
            key = str(sent_id) + "-" +  str(verb_id)
            label = sent_verb_id_to_label.get(key)
            if label != None:
                en_csv_dps.append((dp_id, en_sent, en_verb, label))
                fr_csv_dps.append((dp_id, fr_sent, fr_verb, label))
                verb_id += 1

            else:
                verb_id += 1
        sent_id += 1
    return en_csv_dps, fr_csv_dps


if __name__ == '__main__':
    #prompt_to_csv("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/data.json",
    #              "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/data.csv")

    en_csv_dps, fr_csv_dps = en_fr_to_csv("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_fr_unlabelled/data.json",
                "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/data.json")
    
    print(en_csv_dps[1000:1005])
    print(fr_csv_dps[1000:1005])

    en_train_data, en_eval_data = train_test_split(en_csv_dps, test_size=0.3, random_state=1, shuffle=False)
    fr_train_data, fr_eval_data = train_test_split(fr_csv_dps, test_size=0.3, random_state=1, shuffle=False)

    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/train.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(en_train_data)

    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/eval.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(en_eval_data)

    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_fr_labelled/train.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(fr_train_data)

    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_fr_labelled/eval.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(fr_eval_data)

    print("files written")
