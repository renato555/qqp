import torch
import transformers
from transformers import BertModel, BertTokenizer
from train import QQPModel

transformers.logging.set_verbosity_error()

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    model = QQPModel()
    model.load_state_dict(torch.load("./checkpoints/model_2_1.00_7.pt", map_location=device))
    model.to(device)
    model.eval()

    while True:
        question1 = input("Enter question 1: ")
        question2 = input("Enter question 2: ")

        encoding1 = tokenizer(question1, question2, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs1 = bert_model(**encoding1)

        encoding2 = tokenizer(question2, question1, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs2 = bert_model(**encoding2)

        cls_embeddings = torch.cat((outputs1.last_hidden_state[:, 0, :], outputs2.last_hidden_state[:, 0, :]),
                                dim=1).detach().cpu().numpy()

        with torch.no_grad():
            outputs = model(torch.tensor(cls_embeddings).to(device))
            prediction = torch.sigmoid(outputs).detach().cpu().numpy().item()
            print(f'Probability that they are duplicate: {prediction:.2f}')
            print()
