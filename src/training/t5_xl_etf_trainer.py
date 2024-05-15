from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, \
    T5Config
from datasets import Dataset
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import torch
import os
import json


class T5XLETFTrainer:
    def __init__(self, model_name, etf_dataset,
                 offload_folder="/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/offload"):
        self.model_name = model_name
        self.etf_dataset = etf_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and save model and tokenizer from Hugging Face hub
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        local_model_path = f"/home/oleg/Documents/courses/Stanford/CS224N/FinalProject/code/FolioLLM/models/{model_name.replace('/', '_')}"
        if not os.path.exists(local_model_path):
            os.makedirs(local_model_path, exist_ok=True)

        # Save model and tokenizer locally
        model.save_pretrained(local_model_path)
        self.tokenizer.save_pretrained(local_model_path)

        # Verify if essential tokenizer files are present
        tokenizer_files = os.listdir(local_model_path)
        print(f"Tokenizer files: {tokenizer_files}")

        required_files = ['special_tokens_map.json', 'tokenizer_config.json', 'spiece.model', 'tokenizer.json']
        for file in required_files:
            if file not in tokenizer_files:
                print(f"Missing {file} in {local_model_path}. Attempting to create it.")
                if file == 'special_tokens_map.json':
                    special_tokens_map = self.tokenizer.special_tokens_map
                    with open(os.path.join(local_model_path, file), 'w') as f:
                        json.dump(special_tokens_map, f)
                elif file == 'tokenizer_config.json':
                    tokenizer_config = self.tokenizer.init_kwargs
                    with open(os.path.join(local_model_path, file), 'w') as f:
                        json.dump(tokenizer_config, f)
                elif file == 'spiece.model' or file == 'tokenizer.json':
                    self.tokenizer.save_pretrained(local_model_path)

        config = T5Config.from_pretrained(local_model_path)

        with init_empty_weights():
            self.model = T5ForConditionalGeneration(config)

        device_map = infer_auto_device_map(self.model, max_memory={0: "11GB", "cpu": "32GB"})
        self.model = load_checkpoint_and_dispatch(
            self.model,
            local_model_path,  # Use local path for load_checkpoint_and_dispatch
            device_map=device_map,
            offload_folder=offload_folder
        )

    def tokenize_dataset(self):
        def tokenize_function(examples):
            inputs = examples['prompt']
            targets = examples['expected_answer']
            model_inputs = self.tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
            labels = self.tokenizer(targets, max_length=128, truncation=True, padding='max_length')

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        self.tokenized_dataset = self.etf_dataset.map(tokenize_function, batched=True)

    def train(self):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='no',  # Disable evaluation during training for now
            learning_rate=2e-5,
            per_device_train_batch_size=1,  # Start with batch size of 1
            per_device_eval_batch_size=1,  # Start with batch size of 1
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            fp16=True,  # Enable mixed precision training
            gradient_accumulation_steps=16,  # Accumulate gradients over 16 steps
            optim="adamw_torch",
            gradient_checkpointing=True,  # Enable gradient checkpointing
        )

        trainer = Trainer(
            model=self.model.to(self.device),  # Move model to device
            args=training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
