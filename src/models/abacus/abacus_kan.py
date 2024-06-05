import torch
import torch.nn as nn

from src.models.kan import KAN


class AbacusEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, offset_range):
        super(AbacusEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.offset_range = offset_range
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        batch_size = x.size(0)
        offset = torch.randint(1, self.offset_range + 1, (batch_size, 1)).to(x.device)
        x = x + offset
        return self.embedding(x)


class AbacusKAN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(AbacusKAN, self).__init__()
        self.kan = KAN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

    def forward(self, x):
        return self.kan(x)
