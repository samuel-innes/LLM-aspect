"""
Get the class probabilities for a list of common Russian verbs of motion and plots these in a bar graph.
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

def entropy_from_outputs(outputs):
    #print(outputs)
    softmax = torch.softmax(outputs.logits, 1)[0].tolist()
    #print(torch.softmax(outputs.logits, 1).tolist())
    entropies = [-i*np.log(i) for i in softmax]
    return np.sum(entropies)


class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4}
num_to_class = ["state", "habitual", "activity", "endeavor", "performance"]

tokenizer = AutoTokenizer.from_pretrained('XLM-RoBERTa-base')
model = AutoModelForSequenceClassification.from_pretrained('/home/students/innes/ba2/models/XLM-RoBERTa-BNC', num_labels=5) #XLM-RoBERTa-comtrans-base

max_length = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


model.eval()

verbs_of_motion = [    ["шел", "пошел", "ходил", "сходил"],       # to go (on foot)
    ["ехал", "поехал", "ездил", "съездил"],   # to go (by vehicle)
    ["летел", "полетел", "летал", "слетал"],  # to fly
    ["бежал", "побежал", "бегал", "сбегал"],  # to run
    ["плыл", "поплыл", "плавал", "сплавал"],  # to swim, to sail
    ["нес", "понес", "носил", "сносил"],    # to carry (on foot)
    ["вез", "повез", "возил", "свозил"],    # to transport (by vehicle)
    ["вел", "повел", "водил", "сводил"],    # to lead, to take (someone)
    ["лез", "полезл", "лазил", "слазил"],    # to climb
    ["катил", "покатил", "катал", "скатал"],  # to roll, to wheel
    ["гнал", "погнал", "гонял", "погонял"],     # to chase, to pursue
    [ "тащил", "вытащил", "таскал", "потаскал"],     # to carry, to lug, to drag
    [ "полз", "пополз", "ползал", "поползал"],     # to creep, to crawl
]

label_list = ["endeavor", "performance", "activity", "endeavor"]
all_vms = []
for vm in verbs_of_motion:
    i = 0
    softmaxed_vm = []
    for word in vm:
        text =  word
        label = label_list[i]
        encoding = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            # for i, logit in enumerate(outputs.logits[0]):
            #     print(num_to_class[i], logit.cpu().item())
            #print(outputs.logits)
            #print("entropy: ", entropy_from_outputs(outputs))
            _, predicted = torch.max(outputs.logits, 1)

            print(text.strip())
            print("predicted",num_to_class[predicted.tolist()[0]])
            print("label", label)
            print("---------------------------------")
        i += 1

        softmaxed_vm.append(torch.softmax(outputs.logits, 1)[0].tolist())
    softmaxed_vm = np.array(softmaxed_vm)
    all_vms.append(softmaxed_vm)

all_vms = np.array(all_vms)

tel_impf = all_vms[:, 0]
tel_perf = all_vms[:, 1]
atel_impf = all_vms[:, 2]
atel_perf = all_vms[:, 3]

print(np.mean(tel_impf, axis=0))
print(np.mean(tel_perf, axis=0))
print(np.mean(atel_impf, axis=0))
print(np.mean(atel_perf, axis=0))

tel_impf_mean = np.mean(tel_impf, axis=0)
tel_perf_mean = np.mean(tel_perf, axis=0)
atel_impf_mean = np.mean(atel_impf, axis=0)
atel_perf_mean = np.mean(atel_perf, axis=0)

# plot the mean values
labels = num_to_class
x = np.arange(len(labels))

width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.size':11})

rects1 = ax.bar(x - width*1.5, tel_impf_mean, width, label='Telic Imperfective')
rects2 = ax.bar(x - width*0.5, tel_perf_mean, width, label='Telic Perfective')
rects3 = ax.bar(x + width*0.5, atel_impf_mean, width, label='Atelic Imperfective')
rects4 = ax.bar(x + width*1.5, atel_perf_mean, width, label='Atelic Perfective')

ax.set_xlabel('Class', fontsize=11)
ax.set_ylabel('Mean Probability', fontsize=11)
ax.set_title('Mean Probability of Aspect Class for Motion Verb Type', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.xaxis.label.set_size(11)
ax.yaxis.label.set_size(11)
ax.legend()

# Attach a text label above each bar in *rects*, displaying its height.
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.savefig("/home/students/innes/ba2/LLM-aspect/img/russ_mvs_BNC.jpeg", dpi=300)
plt.show()

