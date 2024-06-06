import torch
import torch.nn as nn
import random
from transformers import AutoTokenizer
from src.models.kan import KAN


class AbacusEmbedding(torch.nn.Module):
    """
    Abacus Embeddings, learned embeddings reused for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    """
    def __init__(self, digit_tokens, embedding_dim, max_seq_length=1024, max_k=99):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)
        self.max_k = max_k

    def helper(self, mask, device):
        mask_shape = mask.shape
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask
        segment_ids = torch.cumsum(starts, dim=1)
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        positions = index - reset_index.gather(1, segment_ids) + 1
        result = positions * mask

        # Debugging information
        # print(f"mask: {mask}")
        # print(f"shifted_mask: {shifted_mask}")
        # print(f"starts: {starts}")
        # print(f"segment_ids: {segment_ids}")
        # print(f"index: {index}")
        # print(f"reset_index: {reset_index}")
        # print(f"positions: {positions}")
        # print(f"result: {result}")


        return result

    def forward(self, input_ids):
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)
        k = 0
        if self.training:
            k = random.randint(0, self.max_k)
            output[output > 0] += k  # as we already have ones in the tensor, the tensor values will be k+1

        # Debugging information
        # print(f"input_ids: {input_ids}")
        # print(f"mask: {mask}")
        # print(f"output: {output}")
        #
        return self.embedding(output)


class AbacusKAN(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, hidden_size3, output_size, digit_tokens,
                 embedding_dim, max_seq_length=1024, max_k=99):
        super(AbacusKAN, self).__init__()
        self.abacus_embedding = AbacusEmbedding(digit_tokens, embedding_dim, max_seq_length, max_k)
        self.fc1 = nn.Linear(embedding_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.abacus_embedding(x)
        batch_size, seq_len, embedding_dim = x.size()
        x = x.view(batch_size * seq_len, embedding_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(batch_size, seq_len, -1)
        x = x.mean(dim=1)
        return x