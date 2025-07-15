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

# write a new python script that calls xfoil, provide file name, and alfa (use aseq command in xfoil). automate the process just now with python that eventually gets cl, 
# and all the cl gets fed for the training dataset (expect output)

# basically preparing the large dataset for the code


#df = pd.read_csv("training_data.txt")  # Make sure this file exists and has the right columns
#
#X_images = np.load("images.npy")   # (N, H, W) numpy array
#X_aoa = df["aoa"].values           # (N,) numpy array or list
#y_cl = df["cl"].values    

# Load training data (space-separated, no header)
df = pd.read_csv("training_data.txt", header=None, sep=',', names=["image_path", "aoa", "cl"])

# Load and stack images
images_file_path = 'C:\\Users\\alexa\\Downloads\\images\\images'
+ "/images/"
#add actual path look up how to do this 
image_arrays = []
for path in df["image_path"]:
    print("Processing image file: ", path)
    if(len(path)) > 20:
        print("namelength > 20, skip...") #filters out long file names (cuz xfoil is unc)
        continue
    if not os.path.exists(os.path.join(images_file_path, path)) :
        print(path," does not exist, skip...")
        continue
    img = Image.open(os.path.join(images_file_path, path)).convert('L')  # Convert to grayscale
    img_array = np.array(img)           # Shape (H, W)
    image_arrays.append(img_array)

X_images = np.stack(image_arrays)  # Shape (N, H, W)
X_aoa = df["aoa"].values           # Shape (N,)
y_cl = df["cl"].values             # Shape (N,)
#take training data


# Normalize AoA (critical enhancement)
aoa_scaler = StandardScaler()
X_aoa_normalized = aoa_scaler.fit_transform(X_aoa.reshape(-1, 1))

# Split with GroupKFold to prevent leakage (enhancement)

groups = df["image_path"].values  # Ensure same airfoil isn't in both sets
# print(groups)
# print(groups.size)
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

        # AoA branch, this is a 1 -> 16 linear mapping from AoA input
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

# Physics-guided loss (critical enhancement)
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

