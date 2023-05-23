import csv
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

with open('data/train.csv', 'r') as input_file, open('data/embeddings.csv', 'w', newline='') as output_file:

    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file)

    csv_writer.writerow(['pair_id', 'is_duplicate', 'cls_embedding'])

    next(csv_reader)

    for row in csv_reader:
        pair_id = row[0]
        qid1 = int(row[1])
        qid2 = int(row[2])
        question1 = row[3]
        question2 = row[4]
        is_duplicate = row[5]

        if qid1 > qid2:
            question1, question2 = question2, question1

        print(f'Current pair: {pair_id}')

        encoding1 = tokenizer(question1, question2, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs1 = model(**encoding1)

        encoding2 = tokenizer(question1, question2, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs2 = model(**encoding2)

        cls_embeddings = torch.cat((outputs1.last_hidden_state[:, 0, :], outputs2.last_hidden_state[:, 0, :]), dim=1).tolist()

        csv_writer.writerow([pair_id, is_duplicate, cls_embeddings])
