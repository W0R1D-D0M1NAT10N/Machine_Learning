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
    Predict lift coefficients for all OBJ files in models folder
    """
    # Initialize predictor
    predictor = LiftCoefficientPredictor(
        model_path='saved_models/model_weights.pth',
        num_points=1024
    )
    
    # Define models folder and AoA range
    models_folder = 'models'
    aoa_range = list(range(0, 16))  # 0 to 15 degrees
    
    print(f"Scanning for OBJ files in '{models_folder}' folder...")
    
    # Check if models folder exists
    if not os.path.exists(models_folder):
        print(f"ERROR: '{models_folder}' folder not found!")
        print("Please create the 'models' folder and place your OBJ files there.")
        return
    
    # Find all OBJ files in models folder
    obj_files = []
    for file in os.listdir(models_folder):
        if file.lower().endswith('.obj'):
            obj_files.append(os.path.join(models_folder, file))
    
    if not obj_files:
        print(f"No OBJ files found in '{models_folder}' folder!")
        return
    
    print(f"Found {len(obj_files)} OBJ file(s):")
    for obj_file in obj_files:
        print(f"  - {os.path.basename(obj_file)}")
    
    print(f"\nPredicting lift coefficients for AoA range: {aoa_range[0]}° to {aoa_range[-1]}°")
    print("=" * 80)
    
    # Process each OBJ file
    all_results = {}
    
    for obj_file in obj_files:
        model_name = os.path.basename(obj_file)
        print(f"\nProcessing: {model_name}")
        print("-" * 40)
        
        # Predict lift coefficients for AoA range
        cl_values = predictor.predict_multiple_aoa(obj_file, aoa_range)
        
        # Store results
        all_results[model_name] = {
            'aoa': aoa_range.copy(),
            'cl': cl_values.copy()
        }
        
        # Display results for this model
        print("AoA (°) | Lift Coefficient")
        print("-" * 25)
        for aoa, cl in zip(aoa_range, cl_values):
            if cl is not None:
                print(f"{aoa:6d} | {cl:12.4f}")
            else:
                print(f"{aoa:6d} | {'ERROR':>12}")
    
    # Create summary plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n{'='*80}")
        print("CREATING LIFT CURVE PLOTS")
        print(f"{'='*80}")
        
        # Create individual plots for each model
        for model_name, data in all_results.items():
            valid_aoas = []
            valid_cls = []
            
            for aoa, cl in zip(data['aoa'], data['cl']):
                if cl is not None:
                    valid_aoas.append(aoa)
                    valid_cls.append(cl)
            
            if valid_aoas:
                plt.figure(figsize=(10, 6))
                plt.plot(valid_aoas, valid_cls, 'bo-', linewidth=2, markersize=6)
                plt.xlabel('Angle of Attack (degrees)', fontsize=12)
                plt.ylabel('Lift Coefficient (CL)', fontsize=12)
                plt.title(f'Lift Curve for {model_name}', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 15)
                
                # Save plot
                plot_filename = f"lift_curve_{model_name.replace('.obj', '')}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Individual plot saved: {plot_filename}")
        
        # Create comparison plot with all models
        if len(all_results) > 1:
            plt.figure(figsize=(12, 8))
            colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, (model_name, data) in enumerate(all_results.items()):
                valid_aoas = []
                valid_cls = []
                
                for aoa, cl in zip(data['aoa'], data['cl']):
                    if cl is not None:
                        valid_aoas.append(aoa)
                        valid_cls.append(cl)
                
                if valid_aoas:
                    color = colors[i % len(colors)]
                    plt.plot(valid_aoas, valid_cls, f'{color}o-', 
                            linewidth=2, markersize=5, 
                            label=model_name.replace('.obj', ''))
            
            plt.xlabel('Angle of Attack (degrees)', fontsize=12)
            plt.ylabel('Lift Coefficient (CL)', fontsize=12)
            plt.title('Lift Curve Comparison - All Models', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 15)
            plt.tight_layout()
            
            comparison_plot = "lift_curve_comparison.png"
            plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Comparison plot saved: {comparison_plot}")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")
    
    # Save results to CSV
    try:
        import pandas as pd
        
        # Create DataFrame for all results
        csv_data = []
        for model_name, data in all_results.items():
            for aoa, cl in zip(data['aoa'], data['cl']):
                csv_data.append({
                    'Model': model_name.replace('.obj', ''),
                    'AoA_degrees': aoa,
                    'Lift_Coefficient': cl
                })
        
        df = pd.DataFrame(csv_data)
        csv_filename = "lift_predictions.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")
        
    except ImportError:
        print("\nPandas not available - skipping CSV export")
    
    print(f"\n{'='*80}")
    print("PREDICTION COMPLETE!")
    print(f"Processed {len(obj_files)} model(s) with {len(aoa_range)} AoA values each")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
