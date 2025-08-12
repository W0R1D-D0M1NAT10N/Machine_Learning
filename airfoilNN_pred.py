import pandas as pd
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Load the validation image paths
val_image_paths = pd.read_csv("val_image_paths.csv")['image_path'].tolist()

# Load the CSV with pre-computed values
df = pd.read_csv("airfoil_data_clean.csv")
df.rename(columns={'file_path': 'image_path'}, inplace=True)

# Filter the DataFrame to only validation image paths
val_df = df[df['image_path'].isin(val_image_paths)].copy()

# Sort by image_path and aoa for consistency
val_df = val_df.sort_values(['image_path', 'aoa']).reset_index(drop=True)

# Load the trained model (assume AirfoilCNN and device are defined elsewhere or import them)
import torch
import torch.nn as nn

class AirfoilCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.conv_output_size = self._get_conv_output_size(input_height, input_width)
        self.aoa_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(self.conv_output_size + 16, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def _get_conv_output_size(self, h, w):
        with torch.no_grad():
            dummy_input = torch.zeros(1,1,h,w)
            output = self.conv_layers(dummy_input)
            return output.shape[1]
    def forward(self, x_img, x_aoa):
        img_features = self.conv_layers(x_img)
        aoa_features = self.aoa_fc(x_aoa)
        combined = torch.cat([img_features, aoa_features], dim=1)
        return self.head(combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_height, image_width = 100, 30  # or set to match your image preprocessing

model = AirfoilCNN(image_height, image_width).to(device)
model.load_state_dict(torch.load("airfoil_cnn.pth"))
model.eval()

import joblib
import os
joblibfile = "aoa_scaler.joblib"
if os.path.exists(joblibfile):
    aoa_scaler = joblib.load("aoa_scaler.joblib")
    print(f"The AoA scalar value: {aoa_scaler.scale_[0]}")
else:
    from sklearn.preprocessing import StandardScaler
    aoa_scaler = StandardScaler()
    aoa_scaler.fit(df['aoa'].values.reshape(-1, 1))
    aoa_scaler.mean_ = np.array([5.88033051])
    aoa_scaler.scale_ = np.array([3.52893337])  # Replace with your actual scale from training

# Prepare image loading (assume images are in the same folder as training)
from PIL import Image
import numpy as np

# Detect OS and set image path accordingly
if os.name == 'nt':
    images_file_path = r"C:\Users\Owner\airfoil_images\images"
else:
    images_file_path = "images"

# Initialize lists to store true and predicted values
true_values = []
predicted_values = []

for idx, row in val_df.iterrows():
    image_path = row['image_path']
    aoa = row['aoa']
    cl_true = row['cl']
    # Load and preprocess image
    image_filename = image_path.replace('.dat', '.png')
    full_image_path = os.path.join(images_file_path, image_filename)
    if not os.path.exists(full_image_path):
        continue  # skip if image is missing
    img = Image.open(full_image_path).convert('L').resize((100, 30))
    img_array = np.array(img, dtype=np.float32)
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    # Normalize AoA
    aoa_norm = aoa_scaler.transform([[aoa]])
    aoa_tensor = torch.tensor(aoa_norm, dtype=torch.float32, device=device)  # (1, 1)
    # Predict
    with torch.no_grad():
        pred = model(img_tensor, aoa_tensor).item()
    true_values.append(cl_true)
    predicted_values.append(pred)

# Calculate the correlation coefficient
correlation = r2_score(true_values, predicted_values)
print(f"Correlation coefficient: {correlation:.4f}")

# Scatter plot
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(true_values, predicted_values, alpha=0.5)
plt.xlabel('True $C_L$')
plt.ylabel('Predicted $C_L$')
plt.title('True vs Predicted $C_L$')
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', label='Ideal')
plt.legend()
plt.tight_layout()
plt.savefig('airfoil_true_vs_pred_scatter.png')
plt.show()
print('Scatter plot saved as airfoil_true_vs_pred_scatter.png')
