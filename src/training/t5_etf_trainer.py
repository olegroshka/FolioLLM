from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset

class T5ETFTrainer:
    def __init__(self, model_name, etf_dataset):
        self.model_name = model_name
        self.etf_dataset = etf_dataset
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

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
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
