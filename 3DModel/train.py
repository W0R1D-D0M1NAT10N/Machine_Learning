import torch
import os
from torch.utils.data import DataLoader
from model import PointNetLiftRegressor
from flightpod_dataset import FlightPodMeshDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load dataset
dataset = FlightPodMeshDataset(
    csv_file="flight_data.csv",
    mesh_dir="data",
    num_points=1024,
    device=device
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model, optimizer, loss
model = PointNetLiftRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Create directory for saving models
os.makedirs("saved_models", exist_ok=True)

# Training loop
best_loss = float('inf')
for epoch in range(100):
    total_loss = 0.0
    for pcd, aoa, cl in dataloader:
        pcd, aoa, cl = pcd.to(device), aoa.to(device), cl.to(device)
        pred = model(pcd, aoa)
        loss = criterion(pred, cl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, 'saved_models/best_model.pth')
        print(f"New best model saved with loss: {best_loss:.4f}")
    
    # Save checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'saved_models/checkpoint_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}")

# Save final model
torch.save({
    'epoch': 100,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, 'saved_models/final_model.pth')

print("Training completed!")
print(f"Best loss achieved: {best_loss:.4f}")
print("Models saved in 'saved_models' directory")

# Save just the model for inference (lighter file)
torch.save(model.state_dict(), 'saved_models/model_weights.pth')
print("Model weights saved for inference")
