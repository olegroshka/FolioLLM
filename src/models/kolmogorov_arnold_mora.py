from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn

class KolmogorovArnoldMoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, hidden_features=128):
        super().__init__()
        self.phi = nn.Linear(in_features, hidden_features)
        self.A = nn.Linear(hidden_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.psi = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = torch.sin(self.phi(x))
        x = self.B(self.A(x)) + self.psi(x)
        return x

class KolmogorovArnoldMoRAModel(nn.Module):
    def __init__(self, model, rank_config, hidden_features=128):
        super().__init__()
        self.model = model
        self.rank_config = rank_config
        self.hidden_features = hidden_features
        self.modify_model_layers()

    def modify_model_layers(self):
        for name, module in self.model.named_modules():
            if 'attention' in name and isinstance(module, nn.Module):
                for param_name, param in module.named_parameters():
                    if param.dim() == 2:  # Ensure we modify only weight matrices
                        in_features = param.shape[1]
                        out_features = param.shape[0]
                        layer_rank = self.rank_config.get(name, {}).get(param_name, 4)
                        mora_layer = KolmogorovArnoldMoRALayer(in_features, out_features, rank=layer_rank, hidden_features=self.hidden_features)
                        # Add detailed debug information
                        print(f"Modifying layer: {name}.{param_name}")
                        print(f"  in_features: {in_features}, out_features: {out_features}, rank: {layer_rank}")
                        setattr(module, f"mora_{param_name}", mora_layer)
                        param.requires_grad = False
                        print(f"  Added MoRA layer to {name}.{param_name}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, rank_config, hidden_features=128, *model_args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        instance = cls(model, rank_config=rank_config, hidden_features=hidden_features)
        return instance

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
