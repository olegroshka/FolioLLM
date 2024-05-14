from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from datasets import Dataset


class ETFTrainer:
    def __init__(self, model_name, etf_dataset):
        self.model_name = model_name
        self.etf_dataset = etf_dataset
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_dataset(self):
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

        self.tokenized_dataset = self.etf_dataset.map(tokenize_function, batched=True)

    def train(self):
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='no',  # Disable evaluation during training for now..
            #evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
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