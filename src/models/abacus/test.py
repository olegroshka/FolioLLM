import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer

from src.models.abacus.abacus_kan import AbacusKAN
from src.models.kan import KAN

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Initialize tokenizer and digit tokens
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
digit_tokens = tokenizer.convert_tokens_to_ids(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Define the base model
class BaseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define the dimensions
input_dim = X_train.shape[1]
hidden_dim = 256
output_dim = 1
hidden_dim1 = 256
hidden_dim2 = 128
hidden_dim3 = 64
embedding_dim = 128

# Train and evaluate the models
models = [
    BaseModel(input_dim, hidden_dim, output_dim).to(device),
    KAN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim).to(device),
    AbacusKAN(hidden_dim1, hidden_dim2, hidden_dim3, output_dim, digit_tokens, embedding_dim).to(device)
]

def train(model, X, y, epochs=150, lr=0.0005):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

def predict(model, X):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        return model(X)

for i in range(3):
    print(f"\nEval iteration: {i+1}:")
    for model in models:
        train(model, X_train, y_train)
        y_pred = predict(model, X_test)
        mse = mean_squared_error(y_test.cpu(), y_pred.cpu())
        print(f"Model: {model.__class__.__name__}, MSE: {mse:.4f}")
