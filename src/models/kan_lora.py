from peft.tuners.lora import LoraLayer
from torch import nn
from src.models.kan import KAN

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        hidden_size1 = max(1, in_features // 4)  #4 Example: quarter of the input features
        hidden_size2 = max(1, in_features // 8)  #8 Example: eighth of the input features
        hidden_size3 = max(1, out_features // 4)  #4 Example: quarter of the output features

        self.kan = KAN(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)

        # Final linear layer to match nn.Linear behavior
        self.final_linear = nn.Linear(out_features, out_features, bias=bias)

    def forward(self, x):
        # Apply KAN network before the final linear layer
        x = self.kan(x)
        return self.final_linear(x)

    @property
    def weight(self):
        return self.final_linear.weight

    @property
    def bias(self):
        return self.final_linear.bias

def patch_update_kan_lora_layer():
    original_update_layer = LoraLayer.update_layer

    def new_update_lora_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False, **kwargs):
        if isinstance(self, LoraLayer):
            self.lora_A[adapter_name] = KANLayer(self.in_features, r)
            self.lora_B[adapter_name] = KANLayer(r, self.out_features)
            self.lora_dropout[adapter_name] = nn.Dropout(p=lora_dropout)  # Ensure the adapter_name exists in lora_dropout
            return self
        return original_update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, **kwargs)

    LoraLayer.update_layer = new_update_lora_layer

