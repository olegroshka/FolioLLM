import torch
from transformers import AutoTokenizer, AutoModel
from src.models.fin_model import FinModel, ModelResult

class AutoNGMModel(FinModel):
    def __init__(self, model_name, max_new_tokens=64):
        self.tokenizer = AutoModel.from_pretrained(model_name)
        self.model = AutoTokenizer.from_pretrained(model_name, output_attentions=True)
        self.max_new_tokens = max_new_tokens

    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=self.max_new_tokens, attention_mask=inputs.attention_mask, pad_token_id=self.tokenizer.eos_token_id)
        predicted_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        with torch.no_grad():
            attention_outputs = self.model(**inputs, output_attentions=True)
            attentions = attention_outputs.attentions

        return ModelResult(predicted_output, attentions)
