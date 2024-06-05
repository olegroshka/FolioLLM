import random
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import faiss
from faiss import write_index, read_index
from peft import PeftModel


ETFS_NUM = 11796

MODEL_NAME = 'FINGU-AI/FinguAI-Chat-v1'
INDEX_PATH = "../../data/etfs.index"
LORA_PATH = '../pipeline/lora_high/FINGU-AI/FinguAI-Chat-v1'

# I don't apply no_grad()
class MultitaskLM(nn.Module):
    def __init__(self, body_path, class_path=None, select_path=None, sel_cfg=sel_cfg, index_path=None, n_features=1024, lora_path=None):
        super(MultitaskLM, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            body_path, output_hidden_states=True
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.class_head = nn.Linear(n_features, 2, bias=False)
        self.select_head = nn.Linear(n_features, ETFS_NUM, bias=False)

        if class_path:
            self.class_head.load_state_dict(torch.load(class_path))
        if select_path:
            self.select_head.load_state_dict(torch.load(select_path))

        self.out = None

        self.index = None
        self.init_index(index_path)
        self.model.eval()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def init_index(self, index_path):
        if index_path != None:
            self.index = read_index(index_path)
            print(f"FAISS index dimensions: {self.index.d}")

    def classify(self, use_prev=False, **kwargs):
        if not use_prev:
            self.out = self.model(**kwargs).hidden_states[-1]
        embedding = self.encode(True)
        return self.class_head(embedding)

    def encode(self, use_prev=False, **kwargs):
        if not use_prev:
            self.out = self.model(**kwargs).hidden_states[-1]
        enc = self.out.mean(-2)
        normalized_enc = enc / torch.norm(enc)
        print(f"Embedding shape: {normalized_enc.shape}")
        return normalized_enc

    def select(self, use_prev=True, use_index=True, **kwargs):
        if not use_prev:
            self.out = self.model(**kwargs).hidden_states[-1]
        embedding = self.encode(True)
        if not use_index:
            scores = self.select_head(embedding)
            return torch.topk(scores[0], 50)[1][:random.randint(3, 8)]
            
        distances, indices = self.index.search(embedding.detach().numpy(), 50)
        return indices[0, :random.randint(3, 8)]


if __name__ == "__main__":
    m = MultitaskLM(MODEL_NAME)
    text = "hello yann lecun"
    tokenizer = AutoTokenizer.from_pretrained(x)
    tokens = tokenizer(text, return_tensors='pt')

    # check if all works
    m.classify(**tokens)
    _ = m.select(**tokens)
