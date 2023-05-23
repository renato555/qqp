import torch
import torch.nn as nn
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained('bert-base-uncased')

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, cls_embedding):
        logits = self.fc(cls_embedding)
        return logits
    
model = Baseline().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())