import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load your ETF data into a DataFrame (using the specified path)
df = pd.read_json('../../data/etf_data_cleaned_v3.json')


# Function to convert percentage strings to floats
def convert_percentage(value):
    if isinstance(value, str) and value.endswith('%'):
        try:
            return float(value.strip('%')) / 100
        except ValueError:
            return None
    return value


# Apply the conversion function to all columns that might contain percentage strings
percentage_columns = ['expense_ratio', 'return_1d', 'ytd_return', 'return_mtd', 'return_3y', 'bid_ask_spread',
                      'nav_trk_error', 'avg_bid_ask_spread']
for column in percentage_columns:
    if column in df.columns:
        df[column] = df[column].apply(convert_percentage)

# Drop rows with any remaining non-numeric values in features or target
df.dropna(subset=percentage_columns + ['class_assets', 'volume_30d', 'flow_1m', 'holdings', 'return_1d'], inplace=True)

# Select features and target
feature_columns = ['class_assets', 'expense_ratio', 'volume_30d', 'flow_1m', 'nav_trk_error', 'holdings', 'return_1d']
features = df[feature_columns].values
target = df['ytd_return'].values

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Convert to tensors and move to the appropriate device
x_train = torch.tensor(features_normalized, dtype=torch.float32).to(device)
y_train = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)


# Define the Kolmogorov-Arnold Network
class KAN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.sin(self.bn1(self.fc1(x)))
        x = torch.sin(self.bn2(self.fc2(x)))
        x = torch.sin(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# Model, loss function, and optimizer
input_size = features_normalized.shape[1]
hidden_size1 = 1024
hidden_size2 = 512
hidden_size3 = 256
output_size = 1
model = KAN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
num_epochs = 35000
patience = 1500  # Number of epochs to wait for improvement before stopping
best_loss = float('inf')
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# Use the trained model for SHAP analysis
model.eval()


# Define a function to pass data through the model
def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()


# Reduce the number of background samples for SHAP
background = shap.sample(features_normalized, 100)  # Use 100 background samples

# Create a SHAP explainer
explainer = shap.KernelExplainer(model_predict, background)

# Compute SHAP values for the training data
shap_values = explainer.shap_values(features_normalized, nsamples=100)

# Ensure shap_values is an array with the correct dimensions
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Reshape SHAP values to remove the extra dimension
shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

# Debugging information
print(f'SHAP values shape: {shap_values.shape}')
print(f'Features normalized shape: {features_normalized.shape}')
print(f'Feature columns: {feature_columns}')

# Visualize the feature importance
shap.summary_plot(shap_values, features_normalized, feature_names=feature_columns)

# Optionally, save the plot
plt.savefig('feature_importance.png')
