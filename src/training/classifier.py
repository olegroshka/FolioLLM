import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# Step 1: Load the dataset
data_path = '../../data/classifier.csv'
df = pd.read_csv(data_path)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Assuming the dataset has columns 'text' and 'label'
df['label'] = df['label'].astype(int)  # Ensure labels are integers

# Step 2: Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


train_dataset = train_dataset
val_dataset = val_dataset

# Set the format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 4: Model Selection
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=df['label'].nunique())

# Step 5: Training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Step 6: Evaluation
trainer.evaluate()
