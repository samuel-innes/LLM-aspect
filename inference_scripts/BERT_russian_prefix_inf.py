"""
Get the class probabilities for a list sentence-verb pairs in Russian, groups these by verb prefix and plots the probabilities in a bar graph.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import csv
import numpy as np
from tqdm import tqdm

class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4}
class AspectPromptDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx]['input'])
        match = re.search(r'\"([^\"]+)\"', self.data[idx]['instruction'])
        if not match:
            print(self.data[idx]['instruction'])
            raise ValueError
        else:
            verb = match.group(1)
        text = text.strip() + " " + verb
        label = class_to_num[self.data[idx]['output']]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class AspectCSVDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx][0])
        verb = str(self.data[idx][1])
        text_and_verb = text.strip() + " " + verb
        label = self.data[idx][2]

        encoding = self.tokenizer.encode_plus(
            text_and_verb,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            #'labels': torch.tensor(label, dtype=torch.long),
            'sentence': text,
            'verb': verb,
            'labels':label,
        }

def entropy_from_outputs(logits):
    #print(logits)
    softmax = torch.softmax(logits, 0).tolist()
    entropies = [-i*np.log(i) for i in softmax]
    return np.sum(entropies)

# Initialize the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('XLM-RoBERTa-base')
model = AutoModelForSequenceClassification.from_pretrained('/home/students/innes/ba2/models/XLM-RoBERTa-BNC', num_labels=5)

# Define parameters
batch_size = 16
max_length = 128

# eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/eval.csv"
# with open(eval_dataset_path, 'r') as fe:
#     eval_data = json.load(fe)

#eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_fr_labelled/eval.csv"
#eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled/llama3_ambig_binary_a1/output.csv"
eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ru_prefix/data.csv"
eval_data = []
with open(eval_dataset_path, 'r') as ft:
    reader = csv.reader(ft)
    for row in reader:
        eval_data.append(row)

save_entropies = True

# Create data loaders
#val_dataset = AspectPromptDataset(eval_data, tokenizer, max_length)
val_dataset = AspectCSVDataset(eval_data, tokenizer, max_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("len val", len(val_dataset))
#exit()
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Validation loop
model.eval()
val_loss = 0
correct_preds = 0
total_preds = 0
predictions, true_labels = [], []
sents_with_entropy = []

prefixes = ['по', 'на', 'при', 'у', 'в', 'вы', 'за', 'под', 'про', 'с', 'раз', 'из', 'об', 'от', 'пере', 'до', 'над', 'недо', 'no-prefix']
prefix_to_prob = {}
for prefix in prefixes:
    prefix_to_prob[prefix] = []
with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        # print(labels)
        outputs = model(input_ids, attention_mask=attention_mask) #labels=labels
        softmax = torch.softmax(outputs.logits, 0).tolist()
        # print(outputs)
        for label, probs in zip(labels, softmax):
            # print(probs)
            # print(label)
            prefix_to_prob[label].append(probs)

    
with open("/home/students/innes/ba2/LLM-aspect/img/probs3.pkl", 'wb') as fout:
    pickle.dump(prefix_to_prob, fout)