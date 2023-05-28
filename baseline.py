import torch
import torch.nn as nn
from dataset_2 import QQPDataset
from sys import argv
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2 * 768, 1)

    def forward(self, cls_embedding):
        logits = self.fc(cls_embedding)
        return logits

def train_model(epochs, dataloader, model, criterion, optimizer, train_path):
    for epoch in range(epochs):

        total_loss = 0

        model.train()

        for inputs, labels in dataloader:
            inputs = torch.from_numpy(inputs).float().to(device)
            labels = torch.tensor([labels], dtype=torch.float).to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

        torch.save(model.state_dict(), f'checkpoints/model_{train_path[9:15]}_{epoch+1}.pt')

def test_model(dataloader, model):
    true_labels = []
    prediction_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = torch.from_numpy(inputs).float().to(device)
            labels = torch.tensor([labels], dtype=torch.float).to(device)

            true_labels += labels.detach().cpu().numpy()

            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))

            prediction_labels += predicted.detach().cpu().numpy()

    print(classification_report(true_labels, prediction_labels))

if __name__ == "__main__":
    train_path = argv[1]
    test_path = argv[2]

    train_dataset = QQPDataset(train_path, 'embeddings')
    test_dataset = QQPDataset(test_path, 'embeddings')

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = Baseline().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 5

    train_model(epochs, train_dataloader, model, criterion, optimizer, train_path)
    test_model(test_dataloader, model)