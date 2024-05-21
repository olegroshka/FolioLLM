import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class KnowledgeAwareMoRALayer(AutoModelForCausalLM):
    def __init__(self, in_features, out_features, rank, knowledge_dim=128):
        super().__init__()
        self.A = nn.Linear(in_features + knowledge_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.knowledge_embedding = nn.Parameter(torch.randn(knowledge_dim))  # Example knowledge embedding

    def forward(self, x):
        knowledge_expanded = self.knowledge_embedding.unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, knowledge_expanded], dim=-1)
        return self.B(self.A(x))

class KnowledgeAwareMoRAModel(AutoModelForCausalLM):
    def __init__(self, model, rank_config, knowledge_dim=128):
        super().__init__()
        self.model = model
        self.rank_config = rank_config
        self.knowledge_dim = knowledge_dim
        for name, module in self.model.named_children():
            if 'attention' in name:
                for param_name, param in module.named_parameters():
                    in_features = param.shape[-1]
                    out_features = param.shape[0]
                    layer_rank = self.rank_config.get(name, {}).get(param_name, 4)
                    mora_layer = KnowledgeAwareMoRALayer(in_features, out_features, rank=layer_rank, knowledge_dim=self.knowledge_dim)
                    setattr(module, f"mora_{param_name}", mora_layer)
                    param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(model, *model_args, **kwargs)

