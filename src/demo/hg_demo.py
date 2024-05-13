import  torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Model

GPT2_LARGE = 'gpt2-large'

MISTRAL_B_V_SHARDED = 'filipealmeida/Mistral-7B-v0.1-sharded'
MICROSOFT_PHI_1_5 = "microsoft/phi-1_5"
LLAMA_B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
FACEBOOK_BART_BASE = 'facebook/bart-base'
T5_SMALL = 't5-small'
BERT_BASE_UNCASED = 'bert-base-uncased'
MISTRAL_B_V = 'mistralai/Mistral-7B-v0.1'


class AttentionScoresDemo:
    def __init__(self, model_name=GPT2_LARGE):
        self.model_name = model_name

        #torch.set_default_device("cuda")
        #model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        #tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
        #self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        # If the model is GPT-2 or similar, use the appropriate class for generation
        if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "openai-gpt"]:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, output_attentions=True)

        self.is_autoregressive_model = isinstance(self.model, (GPT2LMHeadModel, GPT2Model))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token is not None else '[PAD]'

    def get_attention_scores(self, input_text):
        inputs = self.tokenizer(input_text,
                                return_tensors='pt',
                                truncation=True, padding=True,
                                return_attention_mask=True)


        # Check the model type from its configuration
        model_type = self.model.config.model_type

        if model_type in ["gpt2", "openai-gpt", "mistral"]:
            # For GPT-2, Mistral (and similar models)
            attention_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            outputs = self.model.generate(input_ids,
                                          max_new_tokens=64,
                                          attention_mask=attention_mask,
                                          pad_token_id=self.tokenizer.eos_token_id)
            #outputs = self.model.generate(inputs['input_ids'], max_length=100, temperature=1.0, num_return_sequences=1)
            predicted_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extracting attention scores (These models don't provide them by default in the `generate` method)
            # Note: Mistral might not support this out of the box
            with torch.no_grad():
                attention_outputs = self.model(**inputs, output_attentions=True)
                attention = attention_outputs.attentions[-1][0].cpu().detach().numpy()

        else:
            outputs = self.model(**inputs, output_attentions=True)
            attention = outputs.attentions[-1][0].cpu().detach().numpy()
            if hasattr(outputs, "logits"):
                # This is for models that have logits, like T5, GPT-2, etc.
                predicted_token_ids = outputs.logits.argmax(dim=-1).cpu().detach().numpy()
                predicted_output = self.tokenizer.decode(predicted_token_ids[0])
            elif hasattr(outputs, "last_hidden_state"):
                # This is for models like BERT which give embeddings instead of logits
                # Here, you can simply take the embeddings or, if you want a simplified version, take the mean
                embeddings = outputs.last_hidden_state.cpu().detach().numpy()
                predicted_output = embeddings

        print("Input: ", input_text)
        print("Predicted output:", predicted_output)

        return attention


if __name__ == "__main__":
    demo = AttentionScoresDemo()
    input_text = "Do mice catch flies?"
    attention_scores = demo.get_attention_scores(input_text)
    print("attention_scores:", attention_scores)
