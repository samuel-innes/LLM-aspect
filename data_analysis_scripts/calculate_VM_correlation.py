"""This script will calculate the correlation between human judgements of telicity of German verbs of motion"""

import csv
from scipy.stats import pearsonr

class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4, "ambiguous":5}

def read_MV_entropy_file(file_path):
    tuples_list = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        verb_to_entropy = {}
        for i in range(0, len(lines), 9):
            entropy_line = lines[i].strip()
            state_line = lines[i+1].strip()
            habitual_line = lines[i+2].strip()
            activity_line = lines[i+3].strip()
            endeavor_line = lines[i+4].strip() 
            performance_line = lines[i+5].strip()
            verb_line = lines[i + 6].strip()
            class_line = lines[i + 7].strip()
            
            # Extract entropy value
            entropy = round(float(entropy_line.split(': ')[1]), 5)
            
            # Extract verb
            verb = verb_line
            
            # Extract class
            predicted_class = class_line
            
            # Create a tuple and add it to the list
            verb_to_entropy[verb] = (entropy, predicted_class, performance_line, endeavor_line)
            tuples_list.append((verb, entropy, predicted_class, performance_line, endeavor_line))
    
    return verb_to_entropy
    #return tuples_list

def parse_file(file_path):
    results = {}
    with open(file_path, 'r') as file:
        content = file.read()
        entries = content.strip().split('---------------------------------')

        for entry in entries:
            if entry.strip():  # Ensure there is content in the entry
                lines = entry.strip().split('\n')
                state = float(lines[0].split()[1])
                habitual = float(lines[1].split()[1])
                activity = float(lines[2].split()[1])
                endeavor = float(lines[3].split()[1])
                performance = float(lines[4].split()[1])
                entropy = float(lines[5].split()[1])
                verb = lines[6].strip()
                predicted_class = lines[7].strip()
                results[verb] = (entropy, predicted_class, performance, endeavor, state, habitual, activity)
    
    return results

if __name__ == '__main__':

    with open("/home/students/innes/ba2/LLM-aspect/data/HCL_Exp/MoMV_Exp2_3_Analysen_compl.csv", 'r', encoding="utf-8") as fin:
        reader = csv.reader(fin)
        i = 0
        annotation_results = []
        for row in reader:
            if i < 2 or i > 81:
                i+=1
                continue
            verb = row[2]
            directed = row[0]
            local = row[1]
            diff_dir_lok = row[3]
            max_dir = row[4]
            max_lok = row[5]
            diff_lok_temp = row[8]
            if diff_dir_lok != "#VALUE!":
                annotation_results.append((verb, float(directed), float(local), float(diff_dir_lok), float(diff_lok_temp), float(max_dir[:-1]), float(max_lok[:-1])))
            #print(i, verb, directed, local)
            i+=1

    #verb_to_entropy = read_MV_entropy_file("/home/students/innes/ba2/output_files/BERT/german_vm.txt") #MV_BERT_inference
    verb_to_entropy = parse_file("/home/students/innes/ba2/output_files/BERT/german_vm_BNC.txt")
    entropies = []
    annot_diff_dir_lok, annot_diff_lok_temp = [], []
    max_dirs, max_loks = [], []
    performances, activity, endeavor = [], [], []

    locals = []
    directed = []
    for verb in annotation_results:
        entropy = verb_to_entropy[verb[0]][0]
        predicted_class = verb_to_entropy[verb[0]][1]
        entropies.append(entropy)
        performances.append(verb_to_entropy[verb[0]][2])
        endeavor.append(verb_to_entropy[verb[0]][3])

        max_dirs.append(verb[5])
        max_loks.append(verb[6])
        annot_diff_dir_lok.append(verb[3])
        annot_diff_lok_temp.append(verb[4])
        locals.append(verb[2])
        directed.append(verb[1])
    # calculate correlation
    print(pearsonr(entropies, annot_diff_dir_lok))
    print(pearsonr(entropies, annot_diff_lok_temp))
    print(pearsonr(entropies, locals))
    print(pearsonr(entropies, directed))
    print("hi")
    print(pearsonr(performances, locals)) 
    print(pearsonr(performances, directed))
    print(pearsonr(performances, annot_diff_dir_lok))
    print(pearsonr(performances, annot_diff_lok_temp))   
    print("hi")
    print(pearsonr(endeavor, locals))
    print(pearsonr(endeavor, annot_diff_dir_lok))
    print(pearsonr(endeavor, annot_diff_lok_temp))  
    print(pearsonr(endeavor, directed))