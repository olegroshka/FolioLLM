import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoModelForCausalLM


class KAN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.sin(self.bn1(self.fc1(x)))
        x = torch.sin(self.bn2(self.fc2(x)))
        x = torch.sin(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class KolmogorovArnoldLoRAModel(AutoModelForCausalLM):
    def __init__(self, base_model, lora_config, kan_input_dim, kan_hidden_dim1, kan_hidden_dim2, kan_hidden_dim3,
                 kan_output_dim):
        super(AutoModelForCausalLM, self).__init__()
        self.lora_model = get_peft_model(base_model, lora_config)
        self.kan_layer = KAN(kan_input_dim, kan_hidden_dim1, kan_hidden_dim2, kan_hidden_dim3, kan_output_dim)

    def forward(self, input_ids, attention_mask=None):
        lora_output = self.lora_model(input_ids, attention_mask)
        kan_output = self.kan_layer(lora_output.last_hidden_state)
        return kan_output