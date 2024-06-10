import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

from src.models.abacus.abacus_kan_lora import AbacusKANLoRA
from src.models.kan_lora import patch_update_kan_lora_layer

#from src.models.kan_lora import KanLoraModel, patch_update_lora_layer

# Load the text data
with open("data_100.txt", "r") as file:
    lines = file.readlines()

# Preprocess the data into a dictionary
data = {'text': lines}

# Create a Hugging Face Dataset
etf_dataset = Dataset.from_dict(data)

# Split the dataset into train and eval
split_dataset = etf_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("FINGU-AI/FinguAI-Chat-v1")

def tokenize_function(samples):
    tokenized_samples = tokenizer(samples['text'], padding="max_length", truncation=True, max_length=512)
    tokenized_samples["labels"] = tokenized_samples["input_ids"].copy()  # Create labels by copying input_ids
    return tokenized_samples

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Ensure the datasets are in the correct format
tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        #"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
    ]
)

# Load the base FINGU-AI model
base_model = AutoModelForCausalLM.from_pretrained("FINGU-AI/FinguAI-Chat-v1")
base_model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    report_to=["tensorboard"],
    fp16=True,
    fp16_opt_level="O2",
)

# Function to collate data ensuring all required keys are present
def data_collator(features):
    batch = {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'labels': torch.stack([f.get('labels', f['input_ids']) for f in features])
    }
    return batch

# Debug function to check dataset contents
def debug_dataset(dataset, name):
    print(f"Debugging {name} dataset:")
    for i in range(5):
        print(dataset[i])

# Debug train and eval datasets
#debug_dataset(tokenized_train_dataset, "train")
#debug_dataset(tokenized_eval_dataset, "eval")

# Train the base model with vanilla LoRA
vanilla_lora_model = get_peft_model(base_model, lora_config)
vanilla_lora_model.to(device)

vanilla_lora_trainer = Trainer(
    model=vanilla_lora_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator
)
#vanilla_lora_trainer.train()

# Train the base model with Abacus KAN LoRA
digit_tokens = tokenizer.convert_tokens_to_ids(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# kan_lora_model = AbacusKANLoRA(
#     tokenizer=tokenizer,
#     model=base_model,
#     lora_config=lora_config,
#     digit_tokens=digit_tokens,
#     embedding_dim=32,
#     max_seq_length=1024,
#     max_k=99,
#     kan_hidden_dim1=64,
#     kan_hidden_dim2=32,
#     kan_hidden_dim3=16,
#     kan_output_dim=base_model.config.hidden_size
# )

patch_update_kan_lora_layer()
kan_lora_model = get_peft_model(base_model, lora_config)
#kan_lora_model = KanLoraModel(base_model, lora_config, "kan_lora_adapter")
kan_lora_model.to(device)

abacus_kan_lora_trainer = Trainer(
    model=kan_lora_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator
)
abacus_kan_lora_trainer.train()

# Evaluate the models on a test set
test_data = "Sample ETF description text for testing"
test_tokenized_data = tokenizer(test_data, return_tensors="pt", truncation=True, padding=True).to(device)

vanilla_lora_model.eval()
vanilla_lora_test_outputs = vanilla_lora_model(**test_tokenized_data)
vanilla_lora_test_loss = vanilla_lora_test_outputs.loss

kan_lora_model.eval()
abacus_kan_lora_test_outputs = kan_lora_model(**test_tokenized_data)
abacus_kan_lora_test_loss = abacus_kan_lora_test_outputs.loss

# Compare the test losses
print(f"Vanilla LoRA Test Loss: {vanilla_lora_test_loss.items()}")
print(f"Abacus KAN LoRA Test Loss: {abacus_kan_lora_test_loss.items()}")
