import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
import pandas as pd
import json
import re
import csv
import numpy as np
from tqdm import tqdm

# Define your custom dataset class

#class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4}
class_to_num = {"not-ambiguous":0, "ambiguous":1}

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
        text_and_verb = text.strip() + " " + verb
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
            'labels': torch.tensor(label, dtype=torch.long),
            'sentence': text.strip(),
            'verb': verb
        }

class AspectCSVDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx][1])
        verb = str(self.data[idx][2])
        text_and_verb = text.strip() + " " + verb
        #label = class_to_num[self.data[idx][3]]

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
# eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/no_habitual/upsampled_a1/verb_class_train.json"
# with open(eval_dataset_path, 'r') as fe:
#     eval_data = json.load(fe)

eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_unlabelled_bert/output.csv"
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
with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        #labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask) #labels=labels
        #loss = outputs.loss
        #val_loss += loss.item()

        _, predicted = torch.max(outputs.logits, 1)
        # correct_preds += (predicted == labels).sum().item()
        # total_preds += labels.size(0)
        # predictions.extend(predicted.tolist())
        # true_labels.extend(labels.tolist())

        if save_entropies:
            for sent, verb, logits in zip(batch['sentence'], batch['verb'], outputs.logits):
                entropy = entropy_from_outputs(logits)
                sents_with_entropy.append((sent, verb, entropy))

print(len(sents_with_entropy))
print(sents_with_entropy[0])

if save_entropies:
    #with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/no_habitual/upsampled_a1/verb_class_train_entropies_BNC.csv", 'w') as f:
    with open("/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_unlabelled_bert/output_entropies_BNC.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(sents_with_entropy)

# avg_val_loss = val_loss / len(val_loader)
# val_accuracy = correct_preds / total_preds
# val_f1 = f1_score(true_labels, predictions, average='macro')

# print(f'Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}')