"""
This script visualizes the [CLS] embeddings of the fine-tuned BERT model in a 2-dimensional plane.
"""
import torch
import pickle
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

num_to_class = ["state", "habitual", "activity", "endeavor", "performance"]

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('./model_aspect/', num_labels=5)
tokenizer = AutoTokenizer.from_pretrained('XLM-RoBERTa-base')
model = AutoModelForSequenceClassification.from_pretrained('/home/students/innes/ba2/models/XLM-RoBERTa-BNC', num_labels=5)


with open("/home/students/innes/ba2/LLM-aspect/data/multi_lingual_data/german_verbs_of_motion.txt", 'r') as fin:
    words = fin.readlines()
cls_outputs = []
for word in words:
    input_sentence = word

    inputs = tokenizer(input_sentence, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states = True)

    # extract CLS token embeddings
    cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # Get embeddings from last layer

    #probabilities = torch.softmax(outputs.logits, 0).tolist()
    probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy().tolist()[0]  # Assuming logits is of shape [1, num_classes]
    _, predicted = torch.max(outputs.logits, 1)
    #dp_of_prefixes[row['prefix']].append(probabilities)
    #cls_outputs.append((cls_embedding.tolist()[0], num_to_class[row['label']]))
    cls_outputs.append((cls_embedding.tolist()[0],predicted))


embeddings, aspect_label = [], []
for dp in cls_outputs:
    embeddings.append(dp[0]) 
    aspect_label.append(dp[1])
    print(aspect_label)
    #t_label.append(dp[2])
X = np.array(embeddings)

X_embedded = TSNE(n_components=2).fit_transform(X)
#pca = PCA(n_components=2)
#X_embedded = pca.fit_transform(X)

df_embeddings = pd.DataFrame(X_embedded)
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})
df_embeddings = df_embeddings.assign(label=aspect_label)

print(df_embeddings.columns.values)
print(df_embeddings.iloc[0])

import plotly.express as px

fig = px.scatter(
    df_embeddings, x='x', y='y',
    color='label', labels=aspect_label,
    title = 'Fine-tuned BERT aspect embedding space',
    color_discrete_sequence=px.colors.qualitative.Dark2,
    )

fig.write_image("LLM-aspect/img/MVs_BNC2.jpeg",scale=5)
fig.show()

exit()