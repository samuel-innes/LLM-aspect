"""
This script will preprocess the UMC dataset for input to the fine-tuned llama 2 model.
"""
import stanza
import json
from tqdm import tqdm

from preprocess_data import dp_to_prompt

def get_verbs_from_sents(sentences):
    data = []
    id = 0
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)
    for sent in tqdm(sentences):
        if type(sent) == list:
            sent = sent[0]
        dp = {}
        dp["sentence"] = sent.strip()
        doc = nlp(sent)
        if len(doc.sentences) != 1:
            print("issue: ", sent)
            id += 1
            continue
        tagged_sent = doc.sentences[0]
        verbs = []
        for word in tagged_sent.words:
            if word.pos == 'VERB':
                verbs.append([word.text])
        dp["verbs"] = verbs
        dp["id"] = id
        data.append(dp)
        id += 1
    return data


if __name__ == '__main__':
    path = "/home/students/innes/ba2/LLM-aspect/data/umc003-cs-en-ru-triparallel-testset/all/ps2009.toklc.en"
    with open(path, 'r') as f:
        lines = f.readlines()
    
    data = get_verbs_from_sents(lines)
    #data = data[:10]
    prompts = dp_to_prompt(data, True, True, False)
    #prompt_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/umc_unlabelled_ambig_binary/data.json"
    prompt_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/umc_unlabelled/data.json"
    with open(prompt_path, "w") as fout:
        json.dump(prompts, fout)

    print("prompts save to: ", prompt_path)