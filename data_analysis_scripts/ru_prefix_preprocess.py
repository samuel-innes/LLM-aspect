"""
Preprocesses Russian prefix data 
"""

from pymystem3 import Mystem
import csv

def get_longest_prefix(lexeme, gr):
    # Define a list of common Russian prefixes
    prefixes = ['по', 'на', 'при', 'у', 'в', 'вы', 'за', 'под', 'про', 'с', 'раз', 'из', 'об', 'от', 'пере', 'до', 'над', 'недо']
    current_longest_prefix_length = 0
    current_prefix = False
    if 'V' in gr and 'ADV' not in gr:
        current_prefix = "no-prefix"
        for prefix in prefixes:
            if lexeme.startswith(prefix) and len(prefix) > current_longest_prefix_length:
                current_longest_prefix_length = len(prefix)
                current_prefix = prefix            
    return current_prefix


if __name__ == '__main__':
    # load UMC data
    with open("/home/students/innes/ba2/LLM-aspect/data/umc003-cs-en-ru-triparallel-testset/all/ps2009.toklc.ru", 'r') as f:
        data = f.readlines()

    m = Mystem()

    datapoints = []
    for sentence in data:
        sentence = sentence.strip()
        #sentence = "Я надсадил письмо. Она переписала книгу. Мы пройдем по улице."
        analysis = m.analyze(sentence)
        # 
        for word in analysis:
            if 'analysis' in word and word['analysis']:
                lexeme = word['analysis'][0]['lex']
                gr = word['analysis'][0]['gr']
                prefix = get_longest_prefix(lexeme, gr)
                if prefix:
                    datapoints.append((sentence, word['text'], prefix))
                elif prefix== "no-prefix":
                    datapoints.append((sentence, word['text'], "no-prefix"))

    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ru_prefix/data.csv", 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(("sentence", "verb", "prefix"))
        writer.writerows(datapoints)
