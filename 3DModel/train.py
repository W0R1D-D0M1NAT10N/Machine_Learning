import torch
from torch.utils.data import DataLoader
from model import PointNetLiftRegressor
from flightpod_dataset import FlightPodMeshDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Training loop
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

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

