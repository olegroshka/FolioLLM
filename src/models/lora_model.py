import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.original_module(x) + self.B(self.A(x))

class LoRAModel(nn.Module):
    def __init__(self, model, rank=4):
        super().__init__()
        self.model = model
        self.rank = rank
        self.modified_layers = 0
        self.modify_model_layers()

    def get_base_model(self):
        return self.model

    def modify_model_layers(self, module=None, prefix=""):
        if module is None:
            module = self.model

        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and child.weight.requires_grad:
                lora_layer = LoRALayer(child.in_features, child.out_features, rank=self.rank)
                setattr(child, "lora_weight", lora_layer)
                child.weight.requires_grad = False
                self.modified_layers += 1
                print(f"Modified layer: {child_prefix}")
            else:
                self.modify_model_layers(child, child_prefix)

    def forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                label_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_weight"):
                if isinstance(module, nn.Linear):
                    outputs[0] = module(outputs[0]) + module.lora_weight(outputs[0])

        return outputs
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, rank=4, *model_args, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        instance = cls(model, rank=rank)
        return instance


    def __getattr__(self, name):
        if name in ['hf_device_map', 'is_loaded_in_8bit', 'is_loaded_in_4bit', 'is_parallelizable',
                    '_orig_mod', 'is_quantized', 'hf_quantizer', 'quantization_method']:
            return getattr(self.model, name)
        try:
            return super(LoRAModel, self).__getattr__(name)
        except AttributeError:
            return self.model.__getattr__(name)

def main():
    model_name = 'FINGU-AI/FinguAI-Chat-v1'
    model = LoRAModel.from_pretrained(model_name, rank=4)
    print(f"Total modified layers: {model.modified_layers}")

if __name__ == "__main__":
    main()
