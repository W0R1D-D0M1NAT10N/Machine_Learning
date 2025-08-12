from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

## Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
#sphere_mesh = ico_sphere(level=3)
#verts, faces, _ = load_obj("ufo.stl")
#test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
#
## Differentiably sample 5k points from the surface of each mesh and then compute the loss.
#sample_sphere = sample_points_from_meshes(sphere_mesh, 5000)
#sample_test = sample_points_from_meshes(test_mesh, 5000)
#loss_chamfer, _ = chamfer_distance(sample_sphere, sample_test)

import torch
from pytorch3d.io import IO

# Define the device (e.g., CPU or CUDA if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create an IO object
io = IO()

# Specify the path to your STL file
# pytorch3d does not support stl natively. Need to convert to obj format first.
# assimp export ufo.stl ufo.obj
stl_file_path = "ufo.obj"

# Load the mesh from the STL file
try:
    mesh = io.load_mesh(stl_file_path, device=device)
    print(f"Successfully loaded mesh from {stl_file_path}")
    print(f"Number of vertices: {mesh.verts_padded().shape[1]}")
    print(f"Number of faces: {mesh.faces_padded().shape[1]}")
except Exception as e:
    print(f"Error loading STL file: {e}")
