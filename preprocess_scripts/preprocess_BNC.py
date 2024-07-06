"""
convert BNC data into one file (list of sentences)
then so that it is in a llama readable datapoint format
"""

from nltk.corpus.reader.bnc import BNCCorpusReader
#from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.tokenize import sent_tokenize

from umc_verb_extraction import get_verbs_from_sents, dp_to_prompt

import random
import json

if __name__ == '__main__':

    # sents = []
    # aca_bnc_reader = BNCCorpusReader(root="/home/students/innes/ba2/LLM-aspect/data/BNC/Texts/aca", fileids=r'.*\.xml')
    # aca_sents = aca_bnc_reader.sents(aca_bnc_reader.fileids())
    # aca_sents = aca_sents[:10000]
    # for sent in aca_sents:
    #     sents.append([' '.join(sent).strip()])

    # dem_bnc_reader = BNCCorpusReader(root="/home/students/innes/ba2/LLM-aspect/data/BNC/Texts/dem", fileids=r'.*\.xml')
    # dem_sents = dem_bnc_reader.sents(dem_bnc_reader.fileids())
    # dem_sents = dem_sents[:10000]
    # for sent in dem_sents:
    #     sents.append([' '.join(sent).strip()])

    # fic_bnc_reader = BNCCorpusReader(root="/home/students/innes/ba2/LLM-aspect/data/BNC/Texts/fic", fileids=r'.*\.xml')
    # fic_sents = fic_bnc_reader.sents(fic_bnc_reader.fileids())
    # fic_sents = fic_sents[:10000]
    # for sent in fic_sents:
    #     sents.append([' '.join(sent).strip()])

    # news_bnc_reader = BNCCorpusReader(root="/home/students/innes/ba2/LLM-aspect/data/BNC/Texts/news", fileids=r'.*\.xml')
    # news_sents = news_bnc_reader.sents(news_bnc_reader.fileids())
    # news_sents = news_sents[:10000]
    # for sent in news_sents:
    #     sents.append([' '.join(sent).strip()])

    # print(len(sents))

    # with open("/home/students/innes/ba2/LLM-aspect/data/BNC/all_sents.txt", 'w') as fout:
    #     fout.writelines([f"{line}\n" for line in sents])

    # random.shuffle(sents)
    # sents = sents[:10000]
    # data = get_verbs_from_sents(sents)
    # print(len(data))

    # with open("/home/students/innes/ba2/LLM-aspect/data/BNC/dps.json", 'w') as f:
    #     json.dump(data, f)

    with open("/home/students/innes/ba2/LLM-aspect/data/BNC/dps.json", 'r') as f:
        data = json.load(f)


    prompts = dp_to_prompt(data, True, False, False, True)
    #prompt_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/umc_unlabelled_ambig_binary/data.json"
    prompt_path = "/home/students/innes/ba2/LLM-aspect/data/BNC/prompts.json"
    with open(prompt_path, "w") as fout:
        json.dump(prompts, fout)


