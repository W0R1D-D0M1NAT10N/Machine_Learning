import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch.cuda.amp as amp
import pandas as pd
from PIL import Image
import os

# --------------------
# 1. Data Preparation
# --------------------
class AirfoilDataset(Dataset):
    def __init__(self, images, aoas, cls, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
        self.aoas = torch.tensor(aoas, dtype=torch.float32)                 # (N, 1)
        self.cls = torch.tensor(cls, dtype=torch.float32)                   # (N,)
        
    def __len__(self):
        return len(self.cls)
    
    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "aoa": self.aoas[idx],
            "cl": self.cls[idx]
        }

# Load training data
csv_path = r"C:\Users\alexa\airfoil_data.csv"
# Load training data
csv_path = r"C:\Users\alexa\airfoil_data.csv"
df = pd.read_csv(csv_path, sep=',')  # Defaults to header=0, recognizing the first row as header

# Rename 'file_path' to 'image_path' for consistency
df.rename(columns={'file_path': 'image_path'}, inplace=True)

# Force aoa and cl to numeric (handles any stray non-numeric values)
df['aoa'] = pd.to_numeric(df['aoa'], errors='coerce')
df['cl'] = pd.to_numeric(df['cl'], errors='coerce')
df.dropna(subset=['aoa', 'cl'], inplace=True)  # Drop any rows with invalid numeric values

# Load and stack images
images_file_path = r"C:\Users\alexa\Documents\ML\airfoil_images\images"
image_arrays = []
valid_indices = []  # Track valid rows to filter df later

for idx, path in enumerate(df["image_path"]):
    print("Processing image file: ", path)
    if len(path) > 20:
        print("Name length > 20, skipping...")
        continue
    
    # Assume images are named like '2032c.png' (replace '.dat' with '.png'; adjust if extension differs)
    image_filename = path.replace('.dat', '.png')
    full_image_path = os.path.join(images_file_path, image_filename)
    
    if not os.path.exists(full_image_path):
        print(image_filename, " does not exist, skipping...")
        continue
    
    img = Image.open(full_image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)                       # Shape (H, W)
    image_arrays.append(img_array)
    valid_indices.append(idx)

# Filter df to only valid rows
df_filtered = df.iloc[valid_indices].reset_index(drop=True)

# Stack images and extract features/labels
if len(image_arrays) == 0:
    raise ValueError("No valid images found. Check paths and filenames.")
X_images = np.stack(image_arrays)  # Shape (N, H, W)
X_aoa = df_filtered["aoa"].values  # Shape (N,)
y_cl = df_filtered["cl"].values    # Shape (N,)
X_aoa = df_filtered["aoa"].values  # Shape (N,)
y_cl = df_filtered["cl"].values    # Shape (N,)

# Normalize AoA
aoa_scaler = StandardScaler()
X_aoa_normalized = aoa_scaler.fit_transform(X_aoa.reshape(-1, 1))

# Split with GroupKFold to prevent leakage
groups = df_filtered["image_path"].values
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X_images, y_cl, groups))
train_dataset = AirfoilDataset(X_images[train_idx], X_aoa_normalized[train_idx], y_cl[train_idx])
val_dataset = AirfoilDataset(X_images[val_idx], X_aoa_normalized[val_idx], y_cl[val_idx])

# --------------------
# 2. Model Architecture
# --------------------
class AirfoilCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super().__init__()
        # Image branch
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Enhancement: Stabilize training
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        self.conv_output_size = self._get_conv_output_size(input_height, input_width)

        # AoA branch
        self.aoa_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # Combined head
        self.head = nn.Sequential(
            nn.Linear(self.conv_output_size + 16, 64),
            nn.Dropout(0.2),  # Enhancement: Regularization
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

# --------------------
# 3. Training Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_height, image_width = X_images.shape[1], X_images.shape[2]
model = AirfoilCNN(image_height, image_width).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scaler = amp.GradScaler()  # Mixed-precision training

# Physics-guided loss
def physics_loss(outputs, aoas):
    aoas_rad = aoas * (15 * np.pi / 180)  # Scale AoA to ~radians
    thin_airfoil_cl = 2 * np.pi * aoas_rad
    return torch.mean((outputs - thin_airfoil_cl)**2)

# --------------------
# 4. Training Loop
# --------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

for epoch in range(100):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(device)
        aoas = batch["aoa"].to(device)
        cls = batch["cl"].unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        with amp.autocast():  # Mixed precision
            outputs = model(images, aoas)
            data_loss = criterion(outputs, cls)
            p_loss = physics_loss(outputs, aoas)
            loss = data_loss + 0.1 * p_loss  # Weighted physics loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            aoas = batch["aoa"].to(device)
            cls = batch["cl"].unsqueeze(1).to(device)
            outputs = model(images, aoas)
            val_loss += criterion(outputs, cls).item()
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")

# --------------------
# 5. Visualization & Inference
# --------------------
def plot_predictions(model, dataloader, device, n_samples=5):
    model.eval()
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            img = batch["image"][0].cpu().squeeze()
            aoa = batch["aoa"][0].item()
            true_cl = batch["cl"][0].item()
            pred_cl = model(batch["image"].to(device), batch["aoa"].to(device))[0].item()
            
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"AoA={aoa_scaler.inverse_transform([[aoa]])[0][0]:.1f}°\nTrue $C_L$={true_cl:.2f}\nPred $C_L$={pred_cl:.2f}")
            axes[i].axis("off")
    plt.tight_layout()
    plt.show()

plot_predictions(model, val_loader, device)

# Save model
torch.save(model.state_dict(), "airfoil_cnn.pth")

# Print scaler parameters for inference
print("AoA scaler mean:", aoa_scaler.mean_)
print("AoA scaler scale:", aoa_scaler.scale_)