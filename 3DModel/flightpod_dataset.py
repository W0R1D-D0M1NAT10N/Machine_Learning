import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes, load_ply, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import pandas as pd
import os

class FlightPodMeshDataset(Dataset):
    def __init__(self, csv_file, mesh_dir, num_points=1024, device="cpu"):
        self.data = pd.read_csv(csv_file)
        self.mesh_dir = mesh_dir
        self.num_points = num_points
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        obj_file = os.path.join(self.mesh_dir, row['data'])
        aoa = torch.tensor([row['aoa']], dtype=torch.float32)
        cl = torch.tensor([row['cl']], dtype=torch.float32)

        # Load STL mesh
        verts, faces, _ = load_obj(obj_file, device=self.device)
        mesh = Meshes(verts = [verts], faces=[faces.verts_idx])
        #mesh = load_obj(obj_file, device=self.device)
        #mesh = mesh.scale_verts(1.0 / mesh.verts_packed().norm(dim=1).max())  # normalize size
        scale = 1.0/mesh.verts_packed().norm(dim=1).max().item()
        mesh = mesh.scale_verts(scale)

        # Sample points on surface
        pcd = sample_points_from_meshes(mesh, self.num_points)  # (1, N, 3)
        pcd = pcd.squeeze(0)  # (N, 3)

        return pcd, aoa, cl
