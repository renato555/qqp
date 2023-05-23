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

        encoding = tokenizer(question1, question2, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**encoding)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].tolist()

        csv_writer.writerow([pair_id, is_duplicate, cls_embeddings])
