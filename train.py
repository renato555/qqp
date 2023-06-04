import torch
import torch.nn as nn
import os
from dataset_2 import QQPDataset
from sys import argv
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QQPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2 * 768, 4 * 768),
            nn.BatchNorm1d(4 * 768),
            nn.ReLU(),
            nn.Linear(4 * 768, 4 * 768),
            nn.BatchNorm1d(4 * 768),
            nn.ReLU(),
            nn.Linear(4 * 768, 1)
        )

    def forward(self, cls_embedding):
        logits = self.seq(cls_embedding)
        return logits

def train_model(epochs, train_dataloader, validation_dataloader, model, criterion, optimizer, train_path, patience=3):
    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        total_loss = 0

        model.train()

        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        print(f"Loss: {avg_loss}")

        print('Validating model...')
        val_loss = validate_model(validation_dataloader, model, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Validation loss didn't improve for {patience} epochs. Stopping early.")
                break

        print()
        print()

    print('Saving checkpoint...')
    model_name = os.path.basename(train_path)
    torch.save(model.state_dict(), f'checkpoints/model_{model_name[9:15]}_{epoch+1}.pt')


def validate_model(dataloader, model, criterion):
    true_labels = []
    prediction_labels = []
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels += labels.detach().cpu().numpy().tolist()

            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))

            prediction_labels += predicted.detach().cpu().numpy().tolist()

            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(classification_report(true_labels, prediction_labels, digits=5))
    return avg_loss

if __name__ == "__main__":
    train_path = argv[1]
    validation_path = argv[2]
    test_path = argv[3]

    config = {
        'lr': 1e-2,
        'batch_size': 32,
        'epochs': 40,
        'patience': 3
    }

    train_dataset = QQPDataset(train_path, './embeddings/all_embeddings.npy')
    validation_dataset = QQPDataset(validation_path, './embeddings/all_embeddings.npy')
    test_dataset = QQPDataset(test_path, './embeddings/all_embeddings.npy')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = QQPModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_model(config['epochs'], train_dataloader, validation_dataloader, model, criterion, optimizer, train_path, patience=config['patience'])
    print('\nValidating on TEST dataset...')
    validate_model(test_dataloader, model, criterion)