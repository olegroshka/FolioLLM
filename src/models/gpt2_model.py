import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from src.models.fin_model import FinModel, ModelResult

class GPT2NGMModel(FinModel):

    def __init__(self, model_name, max_new_tokens=64):
        super().__init__(model_name)
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens

    def infer(self, prompt_text):
        inputs = self.tokenizer(prompt_text, return_tensors='pt', truncation=True, padding=True)

        outputs = self.model.generate(inputs["input_ids"], max_new_tokens=self.max_new_tokens, attention_mask=inputs["attention_mask"], pad_token_id=self.tokenizer.eos_token_id)
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        with torch.no_grad():
            attention_outputs = self.model(**inputs, output_attentions=True)
            attentions = attention_outputs.attentions

        return ModelResult(prediction, attentions)

