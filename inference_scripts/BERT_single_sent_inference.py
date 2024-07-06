"""
Performs inference on a single sentence
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np

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
model = AutoModelForSequenceClassification.from_pretrained('/home/students/innes/ba2/models/XLM-RoBERTa-BNC', num_labels=5)

max_length = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

text = "Я ходил по магазину около часа. ходил"

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
    for i, logit in enumerate(outputs.logits[0]):
        print(num_to_class[i], logit.cpu().item())
    print(outputs.logits)
    print("entropy: ", entropy_from_outputs(outputs))
    softmax = torch.softmax(outputs.logits, 1)[0].tolist()
    print(softmax)
    _, predicted = torch.max(outputs.logits, 1)

    print(text.strip())
    print(num_to_class[predicted.tolist()[0]])
    print("---------------------------------")
