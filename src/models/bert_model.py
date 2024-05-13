from transformers import AutoTokenizer, AutoModel

from src.models.fin_model import FinModel, ModelResult

class BERTNGMModel(FinModel):

    def __init__(self, model_name):
        super().__init__(model_name)
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)

    def infer(self, prompt_text):
        inputs = self.tokenizer(prompt_text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        # For BERT, taking the mean of embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()
        return ModelResult(embeddings, outputs.attentions)
