import pandas as pd
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader

# Load the validation image paths
val_image_paths = pd.read_csv("val_image_paths.csv")['image_path'].tolist()

# Load the CSV with pre-computed values
df = pd.read_csv("airfoil_data_clean.csv")

# Filter the DataFrame to only validation image paths
val_df = df[df['image_path'].isin(val_image_paths)].copy()

# Sort by image_path and aoa for consistency
val_df = val_df.sort_values(['image_path', 'aoa']).reset_index(drop=True)

# Load the trained model (assume AirfoilCNN and device are defined elsewhere or import them)
model = AirfoilCNN(image_height, image_width).to(device)
model.load_state_dict(torch.load("airfoil_cnn.pth"))
model.eval()

# Prepare AoA normalization (assume aoa_scaler is available or reload its params)
from sklearn.preprocessing import StandardScaler
# If you saved aoa_scaler params, load them here. Otherwise, fit on the full CSV:
aoa_scaler = StandardScaler()
aoa_scaler.fit(df['aoa'].values.reshape(-1, 1))

# Prepare image loading (assume images are in the same folder as training)
import os
from PIL import Image
import numpy as np

# Detect OS and set image path accordingly
if os.name == 'nt':
    images_file_path = r"C:\\Users\\Owner\\airfoil_images\\images"
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
    aoa_tensor = torch.tensor(aoa_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1)
    # Predict
    with torch.no_grad():
        pred = model(img_tensor, aoa_tensor).item()
    true_values.append(cl_true)
    predicted_values.append(pred)

# Calculate the correlation coefficient
correlation = r2_score(true_values, predicted_values)
print(f"Correlation coefficient: {correlation:.4f}")
