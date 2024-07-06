"""
Creates prompts for Llama model from bilingual COMTRANS data
"""

import stanza
import json
from nltk.corpus import comtrans
from tqdm import tqdm
from umc_verb_extraction import get_verbs_from_sents
from preprocess_data import dp_to_prompt

def get_verbs_from_aligned_sents(als):
    data = []
    id = 0
    en_nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', tokenize_no_ssplit=True)
    fr_nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos', tokenize_no_ssplit=True)

    for al_sent in tqdm(als):
        dp = {}
        
        en_doc = en_nlp(" ".join(al_sent.words))
        fr_doc = fr_nlp(" ".join(al_sent.mots))
        dp["sentence"] = " ".join(al_sent.words)
        dp["fr_sentence"] = " ".join(al_sent.mots)
        if len(en_doc.sentences) != 1 or len(fr_doc.sentences) != 1 :
            print("issue: ", al_sent)
            id += 1
            continue
        tagged_en_sent = en_doc.sentences[0]
        tagged_fr_sent = fr_doc.sentences[0]
        verbs = []
        for word in tagged_en_sent.words: # these are the nlp ones
            if word.pos == 'VERB':
                #get french equivalent
                try:
                    en_word_pos = al_sent.words.index(word.text)
                except ValueError:
                    continue
                french_words_pos = [pos[1] for pos in al_sent.alignment if pos[0] == en_word_pos]
                if len(french_words_pos) == 1:
                    for fr_word in tagged_fr_sent.words:
                        if fr_word.text == al_sent.mots[french_words_pos[0]]:
                            fr_nlp_word = fr_word
                            break
                    if fr_nlp_word.pos == 'VERB': # check that en and fr words are both actually verbs
                        verbs.append((word.text, al_sent.mots[french_words_pos[0]]))

        dp["verbs"] = verbs
        dp["id"] = id
        data.append(dp)
        id += 1
    #print(data)
    return data

if __name__ == '__main__':

    # get 4000 examples
    als = comtrans.aligned_sents('alignment-en-de.txt')[:4000]

    # parse sentences to get verbs
    data = get_verbs_from_aligned_sents(als)

    # save this pre-prompt data
    #with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_unlabelled/data.json", "w", encoding='utf-8') as fout:
    #    json.dump(data, fout)

    # create prompts
    # prompts = dp_to_prompt(data, True, False, False, True)
    # print("Number of datapoints", len(prompts))
    # with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_unlabelled/prompts.json", "w") as fout:
    #     json.dump(prompts, fout)
    # print(prompts[:5])