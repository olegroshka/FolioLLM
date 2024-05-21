from transformers import AutoModelForCausalLM
import torch.nn as nn

class MoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.B(self.A(x))

class MoRAModel(nn.Module):
    def __init__(self, model, rank_config):
        super().__init__()
        self.model = model
        self.rank_config = rank_config
        self.modify_model_layers()

    def modify_model_layers(self):
        for name, module in self.model.named_modules():
            if 'attention' in name and isinstance(module, nn.Module):
                for param_name, param in module.named_parameters():
                    if param.dim() == 2:  # Ensure we modify only weight matrices
                        in_features = param.shape[1]
                        out_features = param.shape[0]
                        layer_rank = self.rank_config.get(name, {}).get(param_name, 4)
                        mora_layer = MoRALayer(in_features, out_features, rank=layer_rank)
                        setattr(module, f"mora_{param_name}", mora_layer)
                        param.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, rank_config, *model_args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(model, rank_config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
