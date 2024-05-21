import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class KolmogorovArnoldLoRALayer(AutoModelForCausalLM):
    def __init__(self, in_features, out_features, rank=4, hidden_features=128):
        super().__init__()
        self.phi = nn.Linear(in_features, hidden_features)
        self.A = nn.Linear(hidden_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.psi = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = torch.sin(self.phi(x))
        x = self.B(self.A(x)) + self.psi(x)
        return x

class KolmogorovArnoldLoRAModel(AutoModelForCausalLM):
    def __init__(self, model, rank=4, hidden_features=128):
        super().__init__()
        self.model = model
        self.rank = rank
        self.hidden_features = hidden_features
        for name, module in self.model.named_children():
            if 'attention' in name:
                for param_name, param in module.named_parameters():
                    in_features = param.shape[-1]
                    out_features = param.shape[0]
                    lora_layer = KolmogorovArnoldLoRALayer(in_features, out_features, rank=self.rank, hidden_features=self.hidden_features)
                    setattr(module, f"lora_{param_name}", lora_layer)
                    param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, rank_config, hidden_features=128, *model_args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(model, rank_config=rank_config, hidden_features=hidden_features)
