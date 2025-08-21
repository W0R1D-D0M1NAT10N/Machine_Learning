import torch
import torch.nn as nn
import torch.nn.functional as F

# Use a point net architecture
# Input point cloud is 3 dimensional, default hidden layer dimension is 64
class PointNetLiftRegressor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=1024):
        super().__init__()
        # Multi-layer perceptron 3x64x64
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU()
        )
        # Multi-layer perceptron (hidden_dim/4+1)x64x1
        self.final_mlp = nn.Sequential(
            nn.Linear(int(hidden_dim/4) + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, points, aoa):
        # points: (B, N, 3)
        B, N, _ = points.shape

        x = self.point_mlp(points)  # (B, N, hidden)
        x = torch.max(x, dim=1)[0]  # (B, hidden) global feature

        x = torch.cat([x, aoa], dim=1)  # (B, hidden+1)
        out = self.final_mlp(x)        # (B, 1)
        return out

