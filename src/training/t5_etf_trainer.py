from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class T5ETFTrainer:
    def __init__(self, model_name, etf_dataset):
        self.model_name = model_name
        self.etf_dataset = etf_dataset
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def tokenize_dataset(self):
        def tokenize_function(sample):
            try:
                # Convert the sample dictionary to a string representation
                input_text = f"ETF Ticker: {sample['etf_ticker']}\n"
                input_text += "Features:\n"
                for feature, value in sample['features'].items():
                    input_text += f"{feature}: {value}\n"

                # Tokenize the input text without truncation
                model_inputs = self.tokenizer(input_text, padding='longest')

                # Set the labels to be the same as the input IDs
                model_inputs["labels"] = model_inputs["input_ids"]

                return model_inputs
            except KeyError as e:
                logging.warning(f"Missing key '{e.args[0]}' in sample: {sample}")
                return None

        self.tokenized_dataset = self.etf_dataset.map(tokenize_function, batched=False, remove_columns=self.etf_dataset.column_names)
        self.tokenized_dataset = self.tokenized_dataset.filter(lambda x: x is not None)

    def train(self):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='no',
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            fp16=True,
            #gradient_accumulation_steps=8,
            #gradient_checkpointing=True,
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