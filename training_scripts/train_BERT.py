import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import json
import re
import csv

# Define your custom dataset class

class_to_num = {"state":0, "habitual":1, "activity":2, "endeavor":3, "performance":4}
#class_to_num = {"ambiguous":0, "not-ambiguous":1}

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
        #label = class_to_num[self.data[idx]['output']]
        label = self.data[idx].get('llama_pred')
        if label == None:
            label = class_to_num[self.data[idx]['output']]
        else:
            label = class_to_num[label]

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
        text = str(self.data[idx][1])
        verb = str(self.data[idx][2])
        text = text.strip() + " " + verb
        label = class_to_num[self.data[idx][3]]

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


# Initialize the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base') #google-bert/bert-base-uncased
model = AutoModelForSequenceClassification.from_pretrained('FacebookAI/xlm-roberta-base', num_labels=5)

# Define training parameters
batch_size = 16
max_length = 128
learning_rate = 2e-5
epochs = 15

# Load and preprocess your dataset
#train_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umc_labelled/llama3_ambig_binary_a1/output.json"
train_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/BNC/BNC_labelled.json"
with open(train_dataset_path, 'r') as ft:
    train_data = json.load(ft)

clean_data = [] # not all dps have a llama pred label
for dp in train_data:
    if dp.get('llama_pred') != None:
        clean_data.append(dp)


train_data = clean_data
train_data, eval_data = train_test_split(clean_data, test_size=0.3)

# train_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/train.csv"
# train_data = []
# with open(train_dataset_path, 'r') as ft:
#     reader = csv.reader(ft)
#     for row in reader:
#         train_data.append(row)

# eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/ambiguity_classification/umr/no_habitual/upsampled_a1/verb_class_eval.json"
# with open(eval_dataset_path, 'r') as fe:
#     eval_data = json.load(fe)

# eval_dataset_path = "/home/students/innes/ba2/LLM-aspect/data/preprocessed/aspect_classification/comtrans_en_labelled/eval.csv"
# eval_data = []
# with open(eval_dataset_path, 'r') as ft:
#     reader = csv.reader(ft)
#     for row in reader:
#         eval_data.append(row)

# Create data loaders
train_dataset = AspectPromptDataset(train_data, tokenizer, max_length)
#train_dataset = AspectCSVDataset(train_data, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = AspectPromptDataset(eval_data, tokenizer, max_length)
#val_dataset = AspectCSVDataset(eval_data, tokenizer, max_length)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("len train", len(train_dataset))
print("len val", len(val_dataset))
#exit()
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
current_best_f1 = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for id, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}')

    # Validation loop
    model.eval()
    val_loss = 0
    correct_preds = 0
    total_preds = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_preds / total_preds
    val_f1 = f1_score(true_labels, predictions, average='macro')
    print(f'Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}')

    if val_f1 > current_best_f1:
        current_best_f1 = val_f1
        model.save_pretrained('/home/students/innes/ba2/models/XLM-RoBERTa-BNC')