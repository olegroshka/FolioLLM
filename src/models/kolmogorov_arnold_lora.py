from peft import get_peft_model
from transformers import AutoModelForCausalLM

from src.models.kan import KAN


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