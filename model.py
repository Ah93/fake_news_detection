# model.py
import torch
import torch.nn as nn

class TransformerGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, model_dim=100, num_heads=2, gru_hidden=64, dropout=0.3, num_classes=2):
        super(TransformerGRUClassifier, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.pos_encoder, num_layers=1)
        self.bigru = nn.GRU(embed_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        out, _ = self.bigru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
