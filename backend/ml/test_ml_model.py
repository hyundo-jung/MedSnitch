import torch
from torch import nn
# Neural Network Model
class Model(nn.Module):
    def __init__(self, num_diag_categories, embedding_dim=8):
        super().__init__()
        self.diagnosis_embedding = nn.Embedding(num_embeddings=num_diag_categories, embedding_dim=embedding_dim)

        # Total input size = original features (13) + embedding dim (8) = 21
        self.layers = nn.Sequential(
            nn.Linear(13 + embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_numeric, x_diag_cat):
        # x_diag_cat: [batch_size], long/int dtype
        embedded_diag = self.diagnosis_embedding(x_diag_cat).squeeze(1)  # [batch_size, embedding_dim]
        x = torch.cat([x_numeric, embedded_diag], dim=1)
        return self.layers(x)