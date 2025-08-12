import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from model import PointNetLiftRegressor
import os

class LiftCoefficientPredictor:
    def __init__(self, model_path='saved_models/model_weights.pth', num_points=1024, device=None):
        """
        Initialize the predictor with trained model
        
        Args:
            model_path: Path to saved model weights
            num_points: Number of points to sample from mesh (should match training)
            device: Device to run prediction on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = num_points
        
        # Load model
        self.model = PointNetLiftRegressor().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def load_and_sample_mesh(self, obj_file):
        """
        Load OBJ file and sample points from mesh
        
        Args:
            obj_file: Path to OBJ file
            
        Returns:
            torch.Tensor: Sampled points (num_points, 3)
        """
        if not os.path.exists(obj_file):
            raise FileNotFoundError(f"OBJ file not found: {obj_file}")
        
        # Load mesh
        mesh = load_objs_as_meshes([obj_file], device=self.device)
        
        # Sample points from mesh surface
        points = sample_points_from_meshes(mesh, num_samples=self.num_points)
        
        # Normalize points (same as training)
        # You might need to adjust this based on your training normalization
        verts = mesh.verts_packed()
        scale = 1.0 / verts.norm(dim=1).max().item()
        points = points * scale
        
        return points.squeeze(0)  # Remove batch dimension
    
    def predict_single(self, obj_file, aoa):
        """
        Predict lift coefficient for single mesh and angle of attack
        
        Args:
            obj_file: Path to OBJ file
            aoa: Angle of attack (degrees or radians, depending on training)
            
        Returns:
            float: Predicted lift coefficient
        """
        try:
            # Load and sample mesh
            points = self.load_and_sample_mesh(obj_file)
            
            # Prepare inputs
            points_batch = points.unsqueeze(0)  # Add batch dimension (1, num_points, 3)
            aoa_tensor = torch.tensor([[aoa]], dtype=torch.float32, device=self.device)  # (1, 1)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(points_batch, aoa_tensor)
                cl = prediction.item()
            
            return cl
            
        except Exception as e:
            print(f"Error processing {obj_file}: {str(e)}")
            return None
    
    def predict_batch(self, obj_files, aoas):
        """
        Predict lift coefficients for multiple meshes and angles of attack
        
        Args:
            obj_files: List of OBJ file paths
            aoas: List of angles of attack (same length as obj_files)
            
        Returns:
            list: List of predicted lift coefficients
        """
        if len(obj_files) != len(aoas):
            raise ValueError("Number of OBJ files must match number of AoA values")
        
        results = []
        for obj_file, aoa in zip(obj_files, aoas):
            cl = self.predict_single(obj_file, aoa)
            results.append(cl)
        
        return results
    
    def predict_multiple_aoa(self, obj_file, aoa_list):
        """
        Predict lift coefficients for single mesh at multiple angles of attack
        
        Args:
            obj_file: Path to OBJ file
            aoa_list: List of angles of attack
            
        Returns:
            list: List of predicted lift coefficients
        """
        # Load mesh once
        try:
            points = self.load_and_sample_mesh(obj_file)
        except Exception as e:
            print(f"Error loading {obj_file}: {str(e)}")
            return [None] * len(aoa_list)
        
        results = []
        for aoa in aoa_list:
            try:
                # Prepare inputs
                points_batch = points.unsqueeze(0)  # Add batch dimension
                aoa_tensor = torch.tensor([[aoa]], dtype=torch.float32, device=self.device)
                
                # Predict
                with torch.no_grad():
                    prediction = self.model(points_batch, aoa_tensor)
                    cl = prediction.item()
                
                results.append(cl)
            except Exception as e:
                print(f"Error predicting for AoA {aoa}: {str(e)}")
                results.append(None)
        
        return results


def main():
    """
    Example usage of the predictor
    """
    # Initialize predictor
    predictor = LiftCoefficientPredictor(
        model_path='saved_models/model_weights.pth',
        num_points=1024
    )
    
    # Example 1: Single predictions
    print("=== Single Predictions ===")
    obj_files = [
        'data/aircraft1.obj',
        'data/aircraft2.obj',
        'data/aircraft3.obj'
    ]
    aoas = [5.0, 10.0, 15.0]  # degrees
    
    for obj_file, aoa in zip(obj_files, aoas):
        if os.path.exists(obj_file):
            cl = predictor.predict_single(obj_file, aoa)
            if cl is not None:
                print(f"{obj_file} at {aoa}° AoA: CL = {cl:.4f}")
        else:
            print(f"File not found: {obj_file}")
    
    # Example 2: Batch prediction
    print("\n=== Batch Predictions ===")
    if all(os.path.exists(f) for f in obj_files):
        cl_values = predictor.predict_batch(obj_files, aoas)
        for obj_file, aoa, cl in zip(obj_files, aoas, cl_values):
            if cl is not None:
                print(f"{obj_file} at {aoa}° AoA: CL = {cl:.4f}")
    
    # Example 3: Multiple AoA for single aircraft
    print("\n=== Multiple AoA for Single Aircraft ===")
    single_aircraft = 'data/aircraft1.obj'
    aoa_sweep = [0, 2, 4, 6, 8, 10, 12, 15, 18, 20]  # degrees
    
    if os.path.exists(single_aircraft):
        cl_sweep = predictor.predict_multiple_aoa(single_aircraft, aoa_sweep)
        print(f"Lift curve for {single_aircraft}:")
        for aoa, cl in zip(aoa_sweep, cl_sweep):
            if cl is not None:
                print(f"  AoA {aoa:2d}°: CL = {cl:.4f}")
    
    # Example 4: Create lift curve data
    print("\n=== Creating Lift Curve Data ===")
    if os.path.exists(single_aircraft):
        import matplotlib.pyplot as plt
        
        valid_aoas = []
        valid_cls = []
        
        for aoa, cl in zip(aoa_sweep, cl_sweep):
            if cl is not None:
                valid_aoas.append(aoa)
                valid_cls.append(cl)
        
        if valid_aoas:
            plt.figure(figsize=(10, 6))
            plt.plot(valid_aoas, valid_cls, 'bo-', linewidth=2, markersize=6)
            plt.xlabel('Angle of Attack (degrees)')
            plt.ylabel('Lift Coefficient (CL)')
            plt.title(f'Lift Curve for {os.path.basename(single_aircraft)}')
            plt.grid(True, alpha=0.3)
            plt.savefig('lift_curve.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Lift curve saved as 'lift_curve.png'")


if __name__ == "__main__":
    main()
