import torch.nn as nn
from transformers import AutoModelForCausalLM


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.B(self.A(x))


class LoRAModel(nn.Module):
    def __init__(self, model, rank=4):
        super().__init__()
        self.model = model
        self.rank = rank
        self.modify_model_layers()

    def modify_model_layers(self):
        for name, module in self.model.named_modules():
            if 'attention' in name and isinstance(module, nn.Module):
                for param_name, param in module.named_parameters():
                    if param.dim() == 2:  # Ensure we modify only weight matrices
                        in_features = param.shape[1]
                        out_features = param.shape[0]
                        lora_layer = LoRALayer(in_features, out_features, rank=self.rank)
                        setattr(module, f"lora_{param_name}", lora_layer)
                        param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, rank=4, *model_args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        instance = cls(model, rank=rank)
        return instance

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
