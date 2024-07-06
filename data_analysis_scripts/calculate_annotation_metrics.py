"""
Calculate metrics from the manual annotation
"""

import csv

# load the annotated file
def load_annotated_data(annotated_csv_file):
    with open(annotated_csv_file, "r") as f_ann:
        csvreader_ann = csv.reader(f_ann, delimiter=';') # change delimiter
        header = next(csvreader_ann)
        annotated_dps = []
        for row in csvreader_ann:
            annotated_dps.append(row)

    return annotated_dps

# load the "gold-standard"
def load_labelled_data(labelled_csv_file):
    with open(labelled_csv_file, "r") as f_lab:
        csvreader_lab = csv.reader(f_lab)
        labelled_dps = []
        for row in csvreader_lab:
            labelled_dps.append(row)

    return labelled_dps


def calculate_iaa(a1, a2):
    score = 0
    for dp1, dp2 in zip(a1, a2):
        labels1 = set(dp1)
        labels2 = set(dp2)
        score += len(labels1&labels2)/len(labels1|labels2)
    
    return score/len(a1)

def calculate_at_least_one(a1, a2):
    score = 0
    for dp1, dp2 in zip(a1, a2):
        labels1 = set(dp1)
        labels2 = set(dp2)
        if len(labels1&labels2) > 0:
            score += 1
    
    return score/len(a1)

def ambiguous_iaa(a1,a2):
    score = 0       # both annotators labelled either as ambiguous or not ambiguous
    a1_not_a2 = 0   # dps which a1 labelled as ambiguous, but not a2
    a2_not_a1 = 0
    both_ambig = 0
    for dp1, dp2 in zip(a1, a2):
        if (len(dp1) > 1 and len(dp2) > 1) or len(dp1) == len(dp2):
            score += 1
        if (len(dp1) > 1 and len(dp2) > 1):
            both_ambig += 1
        if (len(dp1) > 1 and len(dp2) == 1):
            a1_not_a2 += 1

        if (len(dp2) > 1 and len(dp1) == 1):
            a2_not_a1 += 1

    print("a1_not_a2", a1_not_a2/len(a1), a1_not_a2)
    print("a1_not_a2 / total ambig a1", 1-(a1_not_a2/(a1_not_a2+score)), a1_not_a2, "/", a1_not_a2+both_ambig)
    print("a2_not_a1", a2_not_a1/len(a1), a2_not_a1)
    print("a2_not_a1 / total ambig a2", 1-(a2_not_a1/(a2_not_a1+score)),a2_not_a1, "/", a2_not_a1+both_ambig)
    return score/len(a1)

if __name__ == '__main__':
    a1_annotated_dps = load_annotated_data("/home/students/innes/ba2/LLM-aspect/data/annotated/SI_annotations.csv")
    a2_annotated_dps = load_annotated_data("/home/students/innes/ba2/LLM-aspect/data/annotated/Sam_annotations.csv")

    labelled_dps = load_labelled_data("/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/upsampling/to_annotate_labelled.csv")

    num_correct_a1,num_correct_a2, num_ambiguous_a1, num_ambiguous_a2, num_unsure, num_ambig_a1_not_h, num_ambig_a2_not_h, num_habit, num_ambig_habit = 0, 0, 0, 0, 0, 0, 0, 0,0
    num_habit_a1, num_habit_a2, num_ambig_habit_a1, num_ambig_habit_a2 = 0,0, 0, 0
    a1_labels_list = []
    a2_labels_list = []
    for a1_dp, a2_dp, gold_dp in zip(a1_annotated_dps,a2_annotated_dps, labelled_dps):
        a1_labels = a1_dp[2].split(",")
        a2_labels = a2_dp[2].split(",")

        a1_labels_list.append(a1_labels)
        a2_labels_list.append(a2_labels)

        gold_label = gold_dp[2]

        if gold_label.upper()[0] in a1_labels: # we only want the first letter of the class from gold label
            num_correct_a1 +=1
        
        if gold_label.upper()[0] in a2_labels:
            num_correct_a2 +=1

        if len(a1_labels) > 1:
            num_ambiguous_a1 += 1
            if 'H' not in a1_labels:
                num_ambig_a1_not_h += 1
        if len(a2_labels) > 1:
            num_ambiguous_a2 += 1
            if 'H' not in a2_labels:
                num_ambig_a2_not_h += 1

        if a1_dp[3] == 'Y':
            num_unsure += 1

        if 'H' in a1_labels:
            num_habit_a1 += 1
            if len(a1_labels) > 1:
                num_ambig_habit_a1 += 1

        if 'H' in a2_labels:
            num_habit_a2 += 1
            if len(a2_labels) > 1:
                num_ambig_habit_a2 += 1

        

    
    print("a1 accuracy: ", num_correct_a1, num_correct_a1/len(a1_annotated_dps))
    print("a2 accuracy: ", num_correct_a2, num_correct_a2/len(a1_annotated_dps))

    print("num ambiguous a1: ", num_ambiguous_a1, num_ambiguous_a1/len(a1_annotated_dps))
    print("num ambiguous a2: ", num_ambiguous_a2, num_ambiguous_a2/len(a2_annotated_dps))
    print("num ambiguous not habitual a1", num_ambig_a1_not_h, num_ambig_a1_not_h/len(a1_annotated_dps))
    print("num ambiguous not habitual a2", num_ambig_a2_not_h, num_ambig_a2_not_h/len(a2_annotated_dps))

    print("num unsure: ", num_unsure, num_unsure/len(a1_annotated_dps))

    print("num habit a1: ", num_habit_a1)
    print("num habit ambiguous a1: ", num_ambig_habit_a1, num_ambig_habit_a1/num_habit_a1)

    print("num habit a2: ", num_habit_a2)
    print("num habit ambiguous a2: ", num_ambig_habit_a2, num_ambig_habit_a2/num_habit_a2)

    score = calculate_iaa(a1_labels_list, a2_labels_list)
    print("IAA", score)
    at_least_one = calculate_at_least_one(a1_labels_list, a2_labels_list)
    print("At least one class the same", at_least_one)
    ambiguous_iaa_score = ambiguous_iaa(a1_labels_list, a2_labels_list)
    print("ambiguous_iaa_score", ambiguous_iaa_score)