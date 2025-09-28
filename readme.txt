Flight Vehicle Surrogate Models

This repository contains code and data pipelines for training neural surrogate models of aerodynamic performance. Two pipelines are included:

1) 2D Airfoil Surrogates using convolutional neural networks (CNNs).
2) 3D Pod Surrogates using a PointNet-style regression model on surface meshes.

The goal is to accelerate conceptual design by predicting lift and drag coefficients from geometry without expensive CFD.

------------------------------------------------------------
Project Structure

- generate_airfoil_data.py      : Run XFOIL across airfoil database to generate CL vs AoA polars
- data_csv_cleanup.py           : Clean up and deduplicate raw XFOIL outputs
- airfoilNN_nvidia.py           : 2D CNN training (airfoil images + AoA → CL)
- airfoilNN_predict.py          : 2D CNN inference and evaluation
- flight_data.csv               : 3D pod dataset (geometry name, AoA, CL, CD, CM)
- model.py                      : PointNetLiftRegressor definition
- train.py                      : Train PointNetLiftRegressor on 3D data
- predict.py                    : Inference for PointNet on OBJ meshes, plots lift curves
- data/, images/, saved_models/, results/ are generated directories

------------------------------------------------------------
Dependencies

- Python 3.9+
- PyTorch (>=1.12, CUDA recommended)
- PyTorch3D
- scikit-learn
- pandas
- numpy
- matplotlib
- Pillow
- tqdm

For CFD data generation (not included here):
- SimFlow / OpenFOAM

Install Python dependencies:

pip install torch torchvision torchaudio
pip install pytorch3d
pip install pandas numpy scikit-learn matplotlib pillow tqdm

------------------------------------------------------------
2D Airfoil Surrogate (CNN)

Data Generation
1) Place .dat airfoil files (UIUC database format) in a folder.
2) Run: python generate_airfoil_data.py
   - Runs XFOIL at Re=1e6, Mach=0.1, AoA 0–12 degrees (step 0.25)
   - Saves results to airfoil_data.csv
3) Run: python data_csv_cleanup.py
   - Produces airfoil_data_clean.csv
4) Convert .dat outlines to .png images (100x30 px grayscale). Store in images/.

Training
python airfoilNN_nvidia.py
- Input: airfoil images + normalized AoA
- Output: CNN model saved as airfoil_cnn.pth, scaler as aoa_scaler.joblib
- Splitting: GroupKFold (5 folds) for family-wise separation
- Hyperparameters: Adam (lr=1.5e-3), batch=32, epochs=1000, MSE loss, optional physics regularization

Prediction & Evaluation
python airfoilNN_predict.py
- Loads model and validation set
- Produces airfoil_predictions_vs_true.csv and airfoil_true_vs_pred_scatter.png
- Reports R2 correlation

Figures
- Figure 2 – CNN performance vs XFOIL (scatter plots, error metrics)

------------------------------------------------------------
3D Pod Surrogate (PointNet)

Dataset
- Training geometries: Gaussian (ufov4), Sech, Sinc
- Validation geometries: Cartoon UFO (spherical disk), Cosine UFO
- Angles of attack: 0–12 degrees in 1 degree steps (some extended to 15, failed 10–11 excluded)
- Labels: CL, CD, CM (training uses CL)
- Source: SimFlow CFD simulations with k–ω SST turbulence model

Training
python train.py
- Loads flight_data.csv and meshes from data/
- Samples 1024 surface points per mesh via PyTorch3D
- Model: PointNetLiftRegressor
  - Per-point MLP: 3→512→512 (ReLU)
  - Global max pooling
  - Concat AoA → MLP 513→128→1
- Optimizer: Adam (lr=1e-3)
- Loss: MSE
- Epochs: 100, batch=4
- Saves weights in saved_models/

Inference
python predict.py
- Loads OBJ meshes from models/
- Runs predictions across AoA sweep (default 0–10 degrees)
- Produces individual lift curve plots and lift_predictions.csv
- Generates comparison plots of all geometries

Figures
- Figure 5 – Lift curves of training geometries (Gaussian, Sech, Sinc)
- Figure 6 – Validation curves on unseen shapes (UFO disk, Cosine UFO)

------------------------------------------------------------
Reproducing Key Figures

- Figure 2: airfoilNN_predict.py → scatter plot of predicted vs true CL
- Figure 5: predict.py with Gaussian/Sech/Sinc models
- Figure 6: predict.py with UFO disk and Cosine UFO

------------------------------------------------------------
Notes

- Ensure XFOIL is installed and accessible on system PATH for 2D data generation
- CFD settings used for 3D data (SimFlow) are documented in Simflow_settings.txt
- Paths in scripts may need adjustment (Windows vs relative)
- GPU strongly recommended for training

------------------------------------------------------------
Citation

If you use this code or data, please cite the accompanying paper:

Alex Yu et al., "Neural Surrogates for Aerodynamic Prediction of 2D Airfoils and 3D Pods", 2025.



Ignore files such as "python refresher" or "airfoilNN_linux" etc. those are test files that serve no use. 
