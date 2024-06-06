import re
import torch
import torch.nn as nn
from peft import LoraConfig, LoraModel, get_peft_model
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.models.abacus.abacus_kan import AbacusEmbedding
from src.models.kan import KAN

class AbacusKANLoRA(nn.Module):
    def __init__(self,
                 tokenizer,
                 model,
                 lora_config,
                 digit_tokens,
                 embedding_dim,
                 max_seq_length,
                 max_k,
                 kan_hidden_dim1,
                 kan_hidden_dim2,
                 kan_hidden_dim3,
                 kan_output_dim):
        super(AbacusKANLoRA, self).__init__()
        self.tokenizer = tokenizer
        self.model = self.model = get_peft_model(model, lora_config)

        self.abacus_embedding = AbacusEmbedding(digit_tokens, embedding_dim, max_seq_length, max_k)

        self.fc1 = nn.Linear(embedding_dim, kan_hidden_dim1)
        self.fc2 = nn.Linear(kan_hidden_dim1, kan_hidden_dim2)
        self.fc3 = nn.Linear(kan_hidden_dim2, kan_hidden_dim3)
        self.fc4 = nn.Linear(kan_hidden_dim3, kan_output_dim)

        # Add the final linear layer
        self.final_layer = nn.Linear(max_seq_length + 1, 1)#kan_output_dim + self.model.config.hidden_size, 1)

    def extract_numerical_tokens(self, input_ids):
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        pattern = r'(\d+)'
        numerical_tokens = [int(match) for match in re.findall(pattern, input_text)]

        if len(numerical_tokens) == 0:
            numerical_tokens = [0]  # Ensure there's at least one token to avoid empty tensor issues

        return torch.tensor(numerical_tokens, device=input_ids.device).unsqueeze(0)

    import torch.nn as nn

    def forward_prev5(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Pass the input through the LoRA layer
        lora_output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                 use_cache=use_cache, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states, return_dict=True)

        # Extract logits from the base model's output
        logits = lora_output.logits

        # Apply Abacus processing on logits
        numerical_tokens = self.extract_numerical_tokens(input_ids)
        numerical_embeddings = self.abacus_embedding(numerical_tokens)

        # Reshape embeddings to match the input dimension of the linear layers
        batch_size, seq_len, embedding_dim = numerical_embeddings.size()
        x = numerical_embeddings.view(batch_size * seq_len, embedding_dim)

        # Pass numerical embeddings through the KAN layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        kan_output = self.fc4(x)

        # Reshape kan_output to match batch and sequence dimensions
        kan_output = kan_output.view(batch_size, seq_len, -1)
        kan_output = kan_output.mean(dim=1)  # Take the mean across the sequence dimension

        # Linear transformation to match the dimensions of logits
        kan_output_dim = kan_output.size(-1)
        logits_dim = logits.size(-1)

        # Dynamically create the linear transformation layer if it doesn't exist
        if not hasattr(self,
                       'kan_to_logits_dim') or self.kan_to_logits_dim.in_features != kan_output_dim or self.kan_to_logits_dim.out_features != logits_dim:
            self.kan_to_logits_dim = nn.Linear(kan_output_dim, logits_dim).to(self.model.device)

        # Transform kan_output to match logits dimensions
        kan_output = self.kan_to_logits_dim(kan_output)

        # Ensure kan_output has the same sequence length and batch size as logits
        kan_output = kan_output.unsqueeze(1).repeat(1, logits.size(1), 1)

        # Combine logits with kan_output by element-wise addition or another appropriate method
        combined_output = logits + kan_output  # Element-wise addition to maintain the same shape

        # Construct the CausalLMOutputWithPast object
        output = CausalLMOutputWithPast(
            logits=combined_output,
            loss=lora_output.loss,
            past_key_values=lora_output.past_key_values,
            hidden_states=lora_output.hidden_states,
            attentions=lora_output.attentions,
        )

        return output

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Pass the input through the LoRA layer
        lora_output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                 use_cache=use_cache, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states, return_dict=True)

        # Extract logits from the base model's output
        logits = lora_output.logits

        # Apply Abacus processing on logits
        numerical_tokens = self.extract_numerical_tokens(input_ids)
        numerical_embeddings = self.abacus_embedding(numerical_tokens)

        # Reshape embeddings to match the input dimension of the linear layers
        batch_size, seq_len, embedding_dim = numerical_embeddings.size()
        x = numerical_embeddings.view(batch_size * seq_len, embedding_dim)

        # Pass numerical embeddings through the KAN layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        kan_output = self.fc4(x)

        # Reshape kan_output to match batch and sequence dimensions
        kan_output = kan_output.view(batch_size, seq_len, -1)
        kan_output = kan_output.mean(dim=1)  # Take the mean across the sequence dimension

        # Ensure kan_output has the same sequence length and batch size as logits
        kan_output = kan_output.unsqueeze(1).repeat(1, logits.size(1), 1)
        kan_output = kan_output.repeat(logits.size(0), 1, 1)

        # Combine logits with kan_output
        combined_output = torch.cat((logits, kan_output), dim=-1)

        # Adjust the input dimension of self.final_layer based on combined_output
        combined_output_dim = combined_output.size(-1)
        # lora_output_dim = logits.size(-1)
        # if not hasattr(self, 'final_layer') or self.final_layer.in_features != combined_output_dim:
        #     self.final_layer = nn.Linear(combined_output_dim, lora_output_dim).to(self.model.device)
        #
        # # Pass the combined output through the final linear layer
        # logits = self.final_layer(combined_output)

        # Construct the CausalLMOutputWithPast object
        output = CausalLMOutputWithPast(
            logits=logits,
            loss=lora_output.loss,
            past_key_values=lora_output.past_key_values,
            hidden_states=lora_output.hidden_states,
            attentions=lora_output.attentions,
        )

        return output

    def forward_prev1(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Pass the input through the LoRA layer
        lora_output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                 use_cache=use_cache, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states, return_dict=True)

        # Access the hidden states from the base model's output
        hidden_states = lora_output.logits

        # Ensure hidden_states is not a scalar and has the expected dimensions
        if hidden_states.dim() == 0:  # If hidden_states is a scalar
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        elif hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)  # Add batch dimension

        # Extract numerical tokens and obtain numerical embeddings
        numerical_tokens = self.extract_numerical_tokens(input_ids)
        numerical_embeddings = self.abacus_embedding(numerical_tokens)

        # Reshape the embeddings to match the input dimension of the linear layers
        batch_size, seq_len, embedding_dim = numerical_embeddings.size()
        x = numerical_embeddings.view(batch_size * seq_len, embedding_dim)

        # Pass the numerical embeddings through the KAN layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        kan_output = self.fc4(x)

        # Reshape kan_output to match the batch and sequence dimensions
        kan_output = kan_output.view(batch_size, seq_len, -1)
        kan_output = kan_output.mean(dim=1)  # Take the mean across the sequence dimension

        # Ensure kan_output has the same sequence length and batch size as hidden_states
        kan_output = kan_output.unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        kan_output = kan_output.repeat(hidden_states.size(0), 1, 1)

        # Combine the LoRA output and the KAN output
        combined_output = torch.cat((hidden_states, kan_output), dim=-1)

        # Extract the dimension of the logits from lora_output to dynamically set the output dimension
        lora_output_dim = lora_output.logits.size(-1)

        # Adjust the input dimension of self.final_layer based on combined_output
        combined_output_dim = combined_output.size(-1)
        # Adjust the input dimension of self.final_layer based on combined_output
        if not hasattr(self, 'final_layer') or self.final_layer.in_features != combined_output_dim:
            self.final_layer = nn.Linear(combined_output_dim, lora_output_dim).to(self.model.device)

        # Pass the combined output through the final linear layer
        logits = self.final_layer(combined_output)

        # Construct the CausalLMOutputWithPast object
        output = CausalLMOutputWithPast(
            logits=logits,
            loss=lora_output.loss,
            past_key_values=lora_output.past_key_values,
            hidden_states=lora_output.hidden_states,
            attentions=lora_output.attentions,
        )

        return output

    def forward_prev0(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Pass the input through the LoRA layer
        lora_output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                 use_cache=use_cache, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states, return_dict=True)

        # Access the hidden states from the base model's output
        hidden_states = lora_output.logits

        # Ensure hidden_states is not a scalar and has the expected dimensions
        if hidden_states.dim() == 0:  # If hidden_states is a scalar
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        elif hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)  # Add batch dimension

        # Debug print: hidden_states shape
        print(f"hidden_states shape: {hidden_states.shape}")

        # Extract numerical tokens and obtain numerical embeddings
        numerical_tokens = self.extract_numerical_tokens(input_ids)
        numerical_embeddings = self.abacus_embedding(numerical_tokens)

        # Reshape the embeddings to match the input dimension of the linear layers
        batch_size, seq_len, embedding_dim = numerical_embeddings.size()
        x = numerical_embeddings.view(batch_size * seq_len, embedding_dim)

        # Pass the numerical embeddings through the KAN layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        kan_output = self.fc4(x)

        # Reshape kan_output to match the batch and sequence dimensions
        kan_output = kan_output.view(batch_size, seq_len, -1)
        kan_output = kan_output.mean(dim=1)  # Take the mean across the sequence dimension

        # Ensure kan_output has the same sequence length and batch size as hidden_states
        kan_output = kan_output.unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        kan_output = kan_output.repeat(hidden_states.size(0), 1, 1)

        # Debug print: kan_output shape
        print(f"kan_output shape: {kan_output.shape}")

        # Combine the LoRA output and the KAN output
        combined_output = torch.cat((hidden_states, kan_output), dim=-1)

        # Adjust the input dimension of self.final_layer based on combined_output
        self.final_layer = nn.Linear(combined_output.size(-1), 1).to(self.model.device)

        # Pass the combined output through the final linear layer
        output = self.final_layer(combined_output)

        # Add a dummy dimension to match the expected output shape
        output = output.unsqueeze(-1)

        return output


    def forward_prev(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        numerical_tokens = self.extract_numerical_tokens(input_ids)
        numerical_embeddings = self.abacus_embedding(numerical_tokens)

        # Reshape the embeddings to match the input dimension of the linear layers
        batch_size, seq_len, embedding_dim = numerical_embeddings.size()
        x = numerical_embeddings.view(batch_size * seq_len, embedding_dim)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        # Reshape back to (batch_size, seq_len, kan_output_dim)
        kan_output = x.view(batch_size, seq_len, -1)
        # Take the mean across the sequence length dimension to get (batch_size, kan_output_dim)
        kan_output = kan_output.mean(dim=1)

        #print(f"kan_output shape after mean: {kan_output.shape}")

        # Pass the remaining arguments to the model, matching Qwen2ForCausalLM forward method
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                             use_cache=use_cache, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=True)

        if return_dict:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        #print(f"hidden_states shape: {hidden_states.shape}")

        # Ensure hidden_states is not a scalar and has the expected dimensions
        if hidden_states.dim() == 0:  # If hidden_states is a scalar
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        elif hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)  # Add batch dimension

        # Ensure kan_output has the same number of dimensions as hidden_states
        kan_output = kan_output.squeeze(1)  # Remove the sequence dimension

        #print(f"hidden_states shape after adjustment: {hidden_states.shape}")
        #print(f"kan_output shape after squeeze: {kan_output.shape}")

        combined_output = torch.cat((hidden_states, kan_output), dim=-1)

        # Pass the combined output through the final linear layer
        output = self.final_layer(combined_output)

        # Squeeze the output to get a scalar value
        output = output.squeeze(-1)

        return output