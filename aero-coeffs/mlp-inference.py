# inference.py - Separate script for loading the model and performing inference

import torch
import torch.nn as nn
import numpy as np
import joblib
import argparse

# Define the model class (must match the one in mlp.py)
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

# Argument parser for input values
parser = argparse.ArgumentParser()
parser.add_argument('--H', type=float, default=100.0, help='Altitude (H)')
parser.add_argument('--V', type=float, default=7492.13, help='Velocity (V)')
parser.add_argument('--alpha', type=float, default=0.0, help='Angle of attack (alpha)')
parser.add_argument('--beta', type=float, default=-90.0, help='Sideslip angle (beta)')
parser.add_argument('--model_path', type=str, default="aero_mlp.pth", help='Path to the saved model .pth file')
args = parser.parse_args()

# Load scalers
scaler_X = joblib.load("scaler_X.joblib")
scaler_y = joblib.load("scaler_y.joblib")

# Load model
model = AeroMLP()
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Prepare input
example_input = np.array([[args.H, args.V, args.alpha, args.beta]])
example_input = scaler_X.transform(example_input)
example_tensor = torch.tensor(example_input, dtype=torch.float32)

# Inference
with torch.no_grad():
    pred = model(example_tensor)
    pred = scaler_y.inverse_transform(pred.numpy())

# Print predictions
print("Predicted Coefficients:")
coeffs = ['CA', 'CN', 'CZ', 'Cll', 'Cnn', 'Cm', 'CD', 'CL']
for coeff, value in zip(coeffs, pred[0]):
    print(f"{coeff}: {value:.5f}")