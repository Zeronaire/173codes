# Edited mlp.py with resume functionality and scaler saving

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import joblib  # For saving scalers

# Argument parser for resume
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--H', type=float, default=100.0, help='Altitude (H)')
parser.add_argument('--V', type=float, default=7492.13, help='Velocity (V)')
parser.add_argument('--alpha', type=float, default=0.0, help='Angle of attack (alpha)')
parser.add_argument('--beta', type=float, default=-90.0, help='Sideslip angle (beta)')
args = parser.parse_args()

# Step 1: Load and Preprocess Data
df = pd.read_csv("aero_small.csv")

# Convert columns to numeric and drop NaNs
numeric_cols = ['H', 'V', 'alpha', 'beta', 'CA', 'CN', 'CZ', 'Cll', 'Cnn', 'Cm', 'CD', 'CL']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Inputs and Outputs
X = df[['H', 'V', 'alpha', 'beta']].values
y = df[['CA', 'CN', 'CZ', 'Cll', 'Cnn', 'Cm', 'CD', 'CL']].values

# Normalize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Save scalers for inference
joblib.dump(scaler_X, "scaler_X.joblib")
joblib.dump(scaler_y, "scaler_y.joblib")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Custom Dataset
class AeroDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AeroDataset(X_train, y_train)
test_dataset = AeroDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Define MLP Model
class AeroMLP(nn.Module):
    def __init__(self):
        super(AeroMLP, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Input: 4 features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)   # Output: 8 coefficients
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AeroMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load checkpoint if resuming
start_epoch = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

# Step 3: Train the Model
for epoch in range(start_epoch, args.num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss / len(train_loader):.6f}")
    
    # Save checkpoint after each epoch
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
torch.save(checkpoint, f"aero_mlp_checkpoint_final.pth")

# Step 4: Evaluate on Test Set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.6f}")

# Step 5: Inference Example
# Example input: H=100, V=7492.13, alpha=0, beta=-90
example_input = np.array([[args.H, args.V, args.alpha, args.beta]])
example_input = scaler_X.transform(example_input)
example_tensor = torch.tensor(example_input, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred = model(example_tensor)
    pred = scaler_y.inverse_transform(pred.numpy())

print("Predicted Coefficients:")
coeffs = ['CA', 'CN', 'CZ', 'Cll', 'Cnn', 'Cm', 'CD', 'CL']
for coeff, value in zip(coeffs, pred[0]):
    print(f"{coeff}: {value:.5f}")

# Save final model
#torch.save(model.state_dict(), "aero_mlp.pth")
