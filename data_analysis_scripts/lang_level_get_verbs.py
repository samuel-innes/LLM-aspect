"""
Gets a list of verbs for the language specified, for entropy calculation later
"""

import nltk
import stanza
import random
from tqdm import tqdm

def get_verbs(path, lang):
    with open(path) as f:
        sents = f.readlines()
    nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos', tokenize_no_ssplit=True)
    verbs = []
    sents = sents[:10000]
    random.shuffle(sents)
    for sent in tqdm(sents):   
        doc = nlp(sent)
        if len(doc.sentences) != 1:
            print("issue: ", sent)
            id += 1
            continue
        tagged_sent = doc.sentences[0]
        
        for word in tagged_sent.words:
            if word.pos == 'VERB':
                verbs.append(word.text + "\n")

    return verbs

if __name__ == '__main__':
    with open("/home/students/innes/ba2/LLM-aspect/data/multi_lingual_data/TED2020/lang_codes.txt", 'r') as f:
        lang_codes = f.readlines()

    lang_codes = ['de']
    for lang in reversed(lang_codes):
        lang_code = lang.strip()
        verbs = get_verbs(f"/home/students/innes/ba2/LLM-aspect/data/multi_lingual_data/TED2020/{lang_code}.tok", lang_code)
        with open(f"/home/students/innes/ba2/LLM-aspect/data/multi_lingual_data/TED2020/{lang_code}.verbs", 'w') as fout:
            fout.writelines(verbs)
        print(len(verbs))

