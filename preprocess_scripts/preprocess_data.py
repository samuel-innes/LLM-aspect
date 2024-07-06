"""
This script will extract verbs from the example sentences in the UMR dataset, along with their labels.
It creates a new file with a list of verbs with their clausal context and aspect label.

The dependency parser used was Stanza (https://stanfordnlp.github.io/stanza/).

The resulting datapoints will be of the format:
[
    {
    "input": "About 200 people were believed killed and 1,500 others were missing in the central Philippines on Friday when a landslide buried an entire village , the Red Cross said .",
    "instruction": "Is the word "said" in this context a state, habitual, activity, endeavor or performance?"
    "output": 
    }
]
"""
import re
import stanza
import penman
import pickle
import json
import csv
import random
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/home/students/innes/ba2/LLM-aspect/data_analysis_scripts')
from calculate_annotation_metrics import load_annotated_data

abbrev_to_full_class_name = {'A': "activity", 'E': "endeavor", 'H':"habitual", 'P':"performance", 'S':"state", "Am": "ambiguous"}

def load_umr_data(filename, n):
    """
    Loads the example UMR annotations.
    Returns a list of dictionaries of the following form:
    {
        "id": "snt1",
        "sent": "200 dead , 1,500 feared missing in Philippines landslide .",
        "UMR": "(s1p / publication-91 ...",
        "alignments":{"s27s": "6-6", ...} 
    }
    """
    data = []
    current_entry = None
    umr_started = False
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("# :: snt"):
                if current_entry:
                    data.append(current_entry)
                current_entry = {'alignments': {}}
                if n != 4:  # this file is formatted differently (no \t)
                    parts = line.split("\t")
                    current_entry['id'] = parts[0].split()[-1]
                    current_entry['sentence'] = " ".join(parts[1:])
                else:
                    parts = line.split(" ")
                    current_entry['id'] = parts[2]
                    current_entry['sentence'] = " ".join(parts[3:])
                
                umr_started = False
            elif line.startswith("# sentence level graph:"):
                current_entry['UMR'] = ""
                umr_started = True
            elif line.startswith("# alignment:"):
                current_entry['alignments'] = {}
                umr_started = False
            elif umr_started:
                current_entry['UMR'] += line + '\n'
            elif line:
                match = re.match(r'(\S+): (\d+-\d+)', line)
                if match:
                    current_entry['alignments'][match.group(1)] = match.group(2)
        if current_entry:
            data.append(current_entry)
    return data

def find_verb_phrases(data):
    """
    Takes the list of data, as described above, and returns a list of verbs in each sentence, together with the alignment:
    Input:
        data (list): see above
    Output:
        output (list): of dicts of the following form:
        {
            "id": "snt1",
            "sentence": "200 dead , 1,500 feared missing in Philippines landslide .",
            "UMR": "(s1p / publication-91 ...",
            "alignments":{"s27s": "6-6", ...},
            "verbs": [("said", 1, "performance"), ...]
        }
    """
    aspect_count = 0
    verbal_aspect_count = 0
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)
    for dp in data:
        sent = dp['sentence']
        #print(dp['UMR'])
        umr = penman.decode(dp['UMR'])
        alignments = dp['alignments']
        doc = nlp(sent)
        if len(doc.sentences) != 1:
            print("issue: ", dp)
            continue
            #raise ValueError
    
        sent = doc.sentences[0]
        verbs = []
        for word in sent.words:
            if word.pos == 'VERB':
                verbal_aspect_count += 1
                print(word.text)

                # find UMR node 
                for i in alignments.keys():
                    if word.id >= int(alignments[i].split('-')[0]) and word.id <= int(alignments[i].split('-')[1]):
                        node = i
                # find aspect class
                for triple in umr.triples:
                    if triple[1] == ':aspect' and triple[0] == node:
                        aspect_class = triple[2]
                        if aspect_class == "partial-affirmative":   # issue with some datapoints
                            continue
                        verbs.append((word.text,word.id,aspect_class))
        
        for triple in umr.triples:
            if triple[1] == ':aspect':
                aspect_count += 1

        dp['verbs'] = verbs
    
    perc = verbal_aspect_count/aspect_count
    print("Total number of verbs: ", verbal_aspect_count)
    print("Total appearances of :aspect parameter: ", aspect_count)
    print(f"Percentage of aspect parameters that are verbal: {perc}%" )
    return data

def load_upsampling_data(path):
    """
    Loads the csv file containing the upsampling data and returns a list of dictionaries with the same fields as the UMR data
    """
    line_count = 0
    data = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            if line_count == 0:
                line_count += 1
                continue

            else:
                dp = {}
                id = "upsmpl" + str(line_count)
                dp["id"] = id
                dp["sentence"] = row[0]
                dp["verbs"] = [(row[1], None, row[2])]
                data.append(dp)

            line_count += 1
    print("len upsampling data", len(data))
    return data


def dp_to_prompt(data, no_labels = False, with_ambiguous_class = False, annotations_present=False, with_habitual=True):
    """
    
    """
    #definitions = "The annotation distinguishes five base level aspectual values — state, habitual, activity, endeavor, and performance. The State value corresponds to stative events: no change occurs during the event. It also includes predicate nominals (be a doctor), predicate locations (be in the forest), and thetic (presentational) possession (have a cat). The Habitual value is annotated on events that occur regularly in the past or present. The Activity value indicates an event has not necessarily ended and may be ongoing at Document Creation Time (DCT). Endeavor is used for processes that end without reaching completion (i.e., termination), whereas Performance is used for processes that reach a completed result state. Event nominals are typically hard to annotate for aspect, since they lack the grammatical cues that verbs often show. Therefore, they are all annotated with the coarse-grained value Process."
    if annotations_present:
        annot_data = load_annotated_data("/home/students/innes/ba2/LLM-aspect/data/annotated/SI_annotations.csv")
    if not with_ambiguous_class:
        # normal description
        definitions = "The annotation distinguishes five base level aspectual values — state, habitual, activity, endeavor, and performance. The State value corresponds to stative events: no change occurs during the event. It also includes predicate nominals (be a doctor), predicate locations (be in the forest), and thetic (presentational) possession (have a cat). The Habitual value is annotated on events that occur regularly in the past or present. The Activity value indicates an event has not necessarily ended and may be ongoing at Document Creation Time (DCT). Endeavor is used for processes that end without reaching completion (i.e., termination), whereas Performance is used for processes that reach a completed result state. Event nominals are typically hard to annotate for aspect, since they lack the grammatical cues that verbs often show. Therefore, they are all annotated with the coarse-grained value Process."
        # long description
        #definitions = "Verbal aspect in language indicates how an action unfolds over time, emphasizing its internal structure, such as whether it is ongoing, completed, repeated, or momentary, distinct from tense which specifies when the action occurs relative to a reference point. The annotation distinguishes five base level aspectual values. The State value corresponds to stative events: no change occurs during the event. The Habitual value is annotated on events that occur regularly. The Activity value indicates an event with no inherent goal that has not necessarily ended and may be ongoing. Endeavor is used for processes which have an inherent end goal but which end without reaching completion (i.e., termination), whereas Performance is used for processes that reach a completed result state."
    
    else:
        # habitual binary
        definitions = "Verbal aspect in language indicates how an action unfolds over time, emphasizing its internal structure, such as whether it is ongoing, completed, repeated, or momentary, distinct from tense which specifies when the action occurs relative to a reference point. Some sentences have multiple possible readings leading to different aspect interpretations."    
    # short desc with habitual
    #definitions = "The State value corresponds to stative events: no change occurs during the event. The Habitual value is annotated on events that occur regularly. The Activity value indicates an event has not necessarily ended and may be ongoing at Document Creation Time. Endeavor is used for processes that end without reaching completion (i.e., termination), whereas Performance is used for processes that reach a completed result state. Sometimes the aspect can be several classes, i.e. ambiguous."
    dp_num = 0
    prompt_data = []
    for dp in data:
        verb_id = -1
        for verb in dp['verbs']:
            verb_id += 1
            if annotations_present:
                annot_dp = annot_data[dp_num]
                annot_labels = annot_dp[2].split(",")
                if not with_habitual:
                    if 'H' in annot_labels:
                        continue
            verb_text = verb[0]
            if with_ambiguous_class:
                question = f"Does \"{verb_text}\" have an ambiguous aspect reading in this sentence (i.e. multiple possible aspect classes)?"
            else:
                question = f"Which class does \"{verb_text}\" belong to in this sentence: state, habitual, activity, endeavor or performance?"
            #question = f"Which class does \"{verb_text}\" belong to in this sentence: state, habitual, activity, endeavor, performance or ambiguous?"
            instruction = definitions + " " + question
            input = dp['sentence']
            if not no_labels:
                output = verb[2]
                if output == "process":
                    continue
                if with_ambiguous_class and annotations_present:
                    if len(annot_labels) > 1: # ambiguous
                        prompt_data.append({"input": input, "instruction":instruction, "output": "ambiguous", "sent_id": dp["id"], "verb_id": verb_id})
                    else:
                        prompt_data.append({"input": input, "instruction":instruction, "output": "not-ambiguous", "sent_id": dp["id"], "verb_id": verb_id})
                    dp_num += 1
                    continue
                else:
                    prompt_data.append({"input": input, "instruction":instruction, "output": output, "sent_id": dp["id"], "verb_id": verb_id})
            else:
                prompt_data.append({"input": input, "instruction":instruction, "sent_id": dp["id"], "verb_id": verb_id})
            dp_num += 1

    return prompt_data
                    

if __name__ == '__main__':
    data = []
    for n in range(1,6):
        path = f"/home/students/innes/ba2/LLM-aspect/data/umr-v1.0/english/english_umr-000{n}.txt"
        data += load_umr_data(path, n)
    print(len(data))

    output = find_verb_phrases(data)
    print(output[0])
    #output = []
    upsampling_data = load_upsampling_data('/home/students/innes/ba2/LLM-aspect/data/upsampling_data/aspect_class_upsampling.csv')
    print(upsampling_data[0])
    output += upsampling_data
    print(output[0])
    print("len output", len(output))
    print("len upsampling", len(upsampling_data))
    pkl_out_path="/home/students/innes/ba2/LLM-aspect/data/preprocessed/umr/upsampling/data.pkl"
    # with open(pkl_out_path, 'wb') as fout:
    #     pickle.dump(output, fout)
    # print("File saved to: ", pkl_out_path)

    # print("Generating TimeLlama style prompts...")
    # prompt_data = dp_to_prompt(output, False, True, True, False)
    # prompt_out_train_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/umr/no_habitual/ambig_binary_a1/verb_class_train.json"
    # prompt_out_eval_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/umr/no_habitual/ambig_binary_a1/verb_class_eval.json"
    
    # train_data, eval_data = train_test_split(prompt_data, test_size=0.3, random_state=1, shuffle=True)
    
    # with open(prompt_out_train_path, 'w') as fout_train:
    #     json.dump(train_data, fout_train)
    # with open(prompt_out_eval_path, 'w') as fout_eval:
    #     json.dump(eval_data, fout_eval)
    
    # print("Prompt train files written to:", prompt_out_train_path )
    # print("Prompt eval files written to:", prompt_out_eval_path )

    