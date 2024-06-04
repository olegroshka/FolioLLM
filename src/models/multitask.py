import random
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


etfs_num = 11796
vocab_size = 151936


def select_scores(
        heatmap, do_sample=False, top_p=0.2, top_k=50, temperature=1,
        sorted_return=True, standardize=False
):
    if standardize: 
        heatmap -= heatmap.mean()
        heatmap /= heatmap.std()

    p = torch.nn.functional.softmax(heatmap, 0)
    val, ind = p.sort(descending=True)
    val, ind = val[:top_k], ind[:top_k]
    m = int((val.cumsum(0) < val.sum()*top_p).sum(dim=-1))

    if not do_sample: return ind[:m], torch.arange(m)

    T = 10/(temperature+1)
    ord_stat = torch.multinomial(val**T, m, replacement=False)

    sorted_indices = ord_stat.sort()[0]
    return ind[sorted_indices], sorted_indices 

# temperature increases randomness of sampling (0,+infty)
sel_cfg = {
    'do_sample': False,  # 
    'temperature': 0,  #
    'top_p': 0.2,
    'top_k': 50,
}


class DownstreamLM(nn.Module):
    def __init__(self, body_path, class_path=None, select_path=None, sel_cfg=sel_cfg):
        super(DownstreamLM, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            body_path, output_hidden_states=True
        )
       
        self.class_head = nn.Linear(1024, 2, bias=False)
        # self.select_head = ...  # something more complicated
        self.select_head = nn.Linear(1024, etfs_num, bias=False)

        if class_path:
            self.class_head.load_state_dict(torch.load(class_path))
        if select_path:
            self.select_head.load_state_dict(torch.load(select_path))

        self.out = None
        self.sel_cfg = sel_cfg

        self.model.eval()

    def forward(self, **kwargs):
        with torch.no_grad():
            return self.model(**kwargs)

    
    def classify(self, **kwargs):
        self.out = self.model(**kwargs).hidden_states[-1]
        out_token = self.out[:,-1]
        return self.class_head(out_token)


    def _heatmap_selection(scores):
        d = 1024
        heatmap / torch.sqrt(d)

    def select(self):
        # X-Attn of out tokens (Batch, Seq, Hid)
        return self.select_head(self.out[:, -1])


model_name = "FINGU-AI/FinguAI-Chat-v1" # output_dir

# m = DownstreamLM(model_name)
# text = "hello yann lecun"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokens = tokenizer(text, return_tensors='pt')

# # verification everything works
# m.classify(**tokens)
# _ = m.select()
