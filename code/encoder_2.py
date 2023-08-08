from transformers import BertModel, BertTokenizer
import torch
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from math import ceil


device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

data = pd.read_csv('../data/all_possible_pairs.csv')

os.makedirs('embeddings', exist_ok=True)

embeddings = []
for i, row in tqdm(data.iterrows(), total=len(data)):
    pair_id = i
    qid1 = int(row['qid1'])
    qid2 = int(row['qid2'])
    question1 = row['question1']
    question2 = row['question2']
    is_duplicate = int(row['is_duplicate'])

    encoding1 = tokenizer(question1, question2, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs1 = model(**encoding1)

    encoding2 = tokenizer(question2, question1, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs2 = model(**encoding2)

    cls_embeddings = torch.cat((outputs1.last_hidden_state[:, 0, :], outputs2.last_hidden_state[:, 0, :]),
                               dim=1).detach().cpu().numpy()

    embeddings.append(cls_embeddings)

    if len(embeddings) == 1000:
        stacked_embeddings = np.concatenate(embeddings, axis=0)
        np.save(f'embeddings/{pair_id + 1}', stacked_embeddings)
        embeddings.clear()

if len(embeddings) > 0:
    stacked_embeddings = np.concatenate(embeddings, axis=0)
    np.save(f'embeddings/{ceil(pair_id/1000) * 1000}', stacked_embeddings)