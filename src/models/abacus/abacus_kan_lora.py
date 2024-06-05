import re
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.models.abacus.abacus_kan import AbacusEmbedding
from src.models.kan import KAN

class AbacusKANLoRA(nn.Module):
    def __init__(self, model, lora_config, num_embeddings, embedding_dim, offset_range, kan_hidden_dim1, kan_hidden_dim2, kan_hidden_dim3, kan_output_dim):
        super(AbacusKANLoRA, self).__init__()
        self.model = get_peft_model(model, lora_config)
        self.abacus_embedding = AbacusEmbedding(num_embeddings, embedding_dim, offset_range)
        self.kan = KAN(embedding_dim, kan_hidden_dim1, kan_hidden_dim2, kan_hidden_dim3, kan_output_dim)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Extract numerical tokens from input_ids
        numerical_tokens = self.extract_numerical_tokens(input_ids)

        # Pass numerical tokens through the Abacus Embedding layer
        numerical_embeddings = self.abacus_embedding(numerical_tokens)

        # Pass the numerical embeddings through the KAN layer
        kan_output = self.kan(numerical_embeddings)

        # Pass the input through the LoRA-adapted model
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Combine the KAN output with the LoRA-adapted model's hidden states
        hidden_states = outputs.last_hidden_state
        combined_output = torch.cat((hidden_states, kan_output), dim=-1)

        return combined_output

    import re

    def extract_numerical_tokens(self, input_ids):
        # Decode the input IDs to get the input text
        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        # Define a regular expression pattern to match numerical values and their labels
        pattern = r'([\w\s]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\%?'

        # Find all matches of the pattern in the input text
        matches = re.findall(pattern, input_text)

        # Create a dictionary to store the numerical tokens and their labels
        numerical_tokens = {}

        # Iterate over the matches and extract the labels and values
        for match in matches:
            label = match[0].strip()
            value = float(match[1])
            numerical_tokens[label] = value

        return numerical_tokens