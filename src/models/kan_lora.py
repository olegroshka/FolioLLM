import math

from peft.tuners.lora import LoraLayer
from torch import nn
from src.models.kan import KAN, KANWithRecurrentLayer, KANGaussianKernel, KANPolynomialKernel, \
    KANGaussianPolynomialCosine, KANReluExp, KANReluTanh, KANGaussianSigmoid, KAND


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        #lora kan is sensitive to the KAN sizing , sizing should be different depending where is applied A or B
        hidden_size1 = max(1, in_features // 1)  #4 Example: quarter of the input features
        hidden_size2 = max(1, in_features // 2)  #8 Example: eighth of the input features
        hidden_size3 = max(1, out_features // 1)  #4 Example: quarter of the output features
        # hidden_size1 = max(64, in_features // 4)  # At least 64, or a quarter of in_features
        # hidden_size2 = max(128, in_features // 2)  # At least 128, or half of in_features
        # hidden_size3 = max(256, in_features // 1)  # At least 256, or equal to in_features

        #self.kan = KAN(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        self.kan = KAND(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        #self.kan = KANWithRecurrentLayer(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        #self.kan = KANGaussianKernel(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        #self.kan = KANPolynomialKernel(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        #self.kan = KANGaussianPolynomialCosine(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        #self.kan = KANReluExp(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        #self.kan = KANReluTanh(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)
        #self.kan = KANGaussianSigmoid(in_features, hidden_size1, hidden_size2, hidden_size3, out_features)

    def forward(self, x):
        # Apply KAN network before the final linear layer
        x = self.kan(x)
        return x #self.final_linear(x)

    @property
    def weight(self):
        return self.kan.fc4.weight #self.final_linear.weight

    @property
    def bias(self):
        return self.kan.fc4.bias#self.final_linear.bias

def patch_update_kan_lora_layer():

    def new_update_lora_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora,
                              use_dora: bool = False, **kwargs):
        # Call the original method to handle the initialization
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # Now replace(!) lora_A and/or lora_B with KANLayer
        #self.lora_A[adapter_name] = KANLayer(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = KANLayer(r, self.out_features, bias=False)
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        #self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            self.pissa_init(adapter_name, init_lora_weights)
        elif init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    LoraLayer.update_layer = new_update_lora_layer
