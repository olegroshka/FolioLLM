
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Read file name from sys.argv or use default
file_path = sys.argv[1] if len(sys.argv) > 1 else '../../data/classifier.csv'
df = pd.read_csv(file_path)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Split texts by labels
texts_label_0 = df[df['label'] == 0]['text'].tolist()
texts_label_1 = df[df['label'] == 1]['text'].tolist()

# Tokenizer setup for future use (not used in current embedding calculation)
model_name = 'FINGU-AI/FinguAI-Chat-v1'
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokens_label_0 = tokenizer(texts_label_0, return_tensors='pt', padding=True, truncation=True)
# tokens_label_1 = tokenizer(texts_label_1, return_tensors='pt', padding=True, truncation=True)

# Calculate embeddings
so = embedding_model.encode(texts_label_0)
sg = embedding_model.encode(texts_label_1)

# Calculate mean of embeddings
som, sgm = torch.tensor(so).mean(0), torch.tensor(sg).mean(0)

# Calculate sentence transformer accuracy
sentr_acc = (((so @ som.numpy()) > (so @ sgm.numpy())).sum() + ((sg @ sgm.numpy()) > (sg @ som.numpy())).sum()) / len(df)
print("Accuracy:", sentr_acc)

# Create a linear layer with the weights
linear_layer = torch.nn.Linear(384, 2)
linear_layer.weight.data = torch.stack((som, sgm))

# Save the state dict
output_file = file_path.split('/')[-1].replace('.csv', '.pth')
torch.save(linear_layer.state_dict(), output_file)

print(f"Model weights saved to {output_file}")
