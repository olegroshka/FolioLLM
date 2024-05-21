import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class KnowledgeAwareLoRALayer(AutoModelForCausalLM):
    def __init__(self, in_features, out_features, rank=4, knowledge_dim=128):
        super().__init__()
        self.A = nn.Linear(in_features + knowledge_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.knowledge_embedding = nn.Parameter(torch.randn(knowledge_dim))  # Example knowledge embedding

    def forward(self, x):
        knowledge_expanded = self.knowledge_embedding.unsqueeze(0).expand(x.size(0), -1)
        x = torch.cat([x, knowledge_expanded], dim=-1)
        return self.B(self.A(x))

class KnowledgeAwareLoRAModel(AutoModelForCausalLM):
    def __init__(self, model, rank=4, knowledge_dim=128):
        super().__init__()
        self.model = model
        self.rank = rank
        self.knowledge_dim = knowledge_dim
        for name, module in self.model.named_children():
            if 'attention' in name:
                for param_name, param in module.named_parameters():
                    in_features = param.shape[-1]
                    out_features = param.shape[0]
                    lora_layer = KnowledgeAwareLoRALayer(in_features, out_features, rank=self.rank, knowledge_dim=self.knowledge_dim)
                    setattr(module, f"lora_{param_name}", lora_layer)
                    param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(model, *model_args, **kwargs)
