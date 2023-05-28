import torch
import torch.nn as nn
from transformers import BertModel
from dataset import QQPDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained('bert-base-uncased')

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2 * bert_model.config.hidden_size, 1)

    def forward(self, cls_embedding):
        logits = self.fc(cls_embedding)
        return logits
    
model = Baseline().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
dataloader = QQPDataset('data/train_light.csv', 'embeddings')

epochs = 5

def train_model(epochs, dataloader):
    for epoch in range(epochs):
        total_loss = 0

        running_loss = 0.0

        model.train()

        for i, (inputs, labels) in enumerate(dataloader):
            inputs = torch.from_numpy(inputs).float().to(device)
            labels = torch.tensor([labels], dtype=torch.float).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels.float())

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

def test_model(dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = torch.from_numpy(inputs).float().to(device)
            labels = torch.tensor([labels], dtype=torch.float).to(device)

            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on test data: {accuracy}%')

train_model(epochs, dataloader)
test_model(dataloader)