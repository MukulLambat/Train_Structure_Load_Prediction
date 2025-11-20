# Train Structure Load Prediction

## Overview
This repository implements a machine-learning pipeline to predict applied loads (force) on train structures from sensor or simulation data. The code covers data loading and preprocessing, training multiple regression models (Linear Regression, SGD, SVM, LightGBM and a PyTorch 1D-CNN), evaluation with common regression metrics, and inference using saved models.

## Key features
- End-to-end pipeline: data preprocessing → model training → evaluation → inference  
- Multiple model implementations:
  - Linear Regression (sklearn)
  - SGDRegressor (sklearn)
  - Support Vector Regressor (sklearn)
  - LightGBM Regressor (lightgbm)
  - 1D Convolutional Neural Network (PyTorch)
- Evaluation metrics and logging: MAE, MAPE, MSE, RMSE, R², Explained Variance

- Inference scripts for sklearn/LightGBM and the CNN
- Example SLURM script for GPU training (Bash_Script.sh)

## Project structure (high level)
- Models/  
  - LGBMRegressor_Model.py  
  - Linear_Regression_Model.py  
  - Stochastic_Gradient_Descent_Model.py  
  - Support_Vector_Machine.py  
  - CNN.py
- Process_Data/  
  - Data_PreProcessing.py
- Trained_Model/ — saved model files (.pkl, .pth)
- Results/ and Code/Results/ — metrics, logs, plots
- Load_Trained_Model_For_Prediction.py — sklearn/LightGBM inference
- CNN_Prediction.py — CNN inference
- Bash_Script.sh — example SLURM job (uses conda env Train_Regression)

## Installation
Recommended: conda environment (repository references env name `Train_Regression`).

1. Install Miniconda/Anaconda (if needed).
2. Create and activate environment:
```bash
conda create -n Train_Regression python=3.8 -y
conda activate Train_Regression
```
3. Install dependencies (pip example):
```bash
pip install requirements.txt
```

## Preparing data
- Place CSV input files in a directory referenced by the preprocessing script. Example used in code:
  /Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Test_data/Combined_Simu_1_accel.csv
- The preprocessing module is Process_Data/Data_PreProcessing.py. Model scripts import it (import Data_PreProcessing as DP). Typical outputs provided by preprocessing: DP.X_Train, DP.Y_Train, DP.X_Test, DP.Y_Test and fitted scalers.

## How to run training
Run any of the model scripts (each imports preprocessing):

LightGBM:
```bash
conda activate Train_Regression
python Models/LGBMRegressor_Model.py
```

Linear Regression:
```bash
python Models/Linear_Regression_Model.py
```

SGD:
```bash
python Models/Stochastic_Gradient_Descent_Model.py
```

Support Vector Regressor:
```bash
python Models/Support_Vector_Machine.py
```

Train / evaluate the CNN (PyTorch):
```bash
python Models/CNN.py
# or submit the SLURM job (edit Bash_Script.sh if needed)
sbatch Bash_Script.sh
```

## How to run inference
- sklearn / LightGBM model:
```bash
python Load_Trained_Model_For_Prediction.py
```
This script:
- Loads a CSV input (path inside the script)
- Loads a saved model (example path: /Users/mukul/Desktop/DLR_Internship/Trained_Model/LGBMRegressor.pkl)
- Applies the same scaler(s) used during training and saves/plots predictions

- CNN inference:
```bash
python CNN_Prediction.py
```

## Expected inputs / outputs
Inputs:
- Tabular CSV files with the same columns expected by Data_PreProcessing.py.
- Typical preprocessing uses 'Force_Applied' as the target and drops ['Time_Step', 'Force_Applied'] when constructing features.

Outputs:
- Saved trained models in Trained_Model/ (examples in repo: LGBMRegressor.pkl, Linear_Regression.pkl, CNN .pth)
- Evaluation metrics, logs and plots saved to Code/Results/ or Results/Validation_Results/
- Text files with metrics, per-epoch logs for CNN, and validation plots

## Dependencies / technologies used
- Python 3.8+  
- numpy, pandas, scikit-learn, matplotlib, seaborn, joblib  
- lightgbm  
- PyTorch (torch, torchvision)  
- sktime (if used by preprocessing)  
- Optional: conda for environment management

## How to reproduce results
1. Update file paths in Process_Data/Data_PreProcessing.py and inference scripts to point to your data.  
2. Ensure conda environment has required packages.  
3. Run a model training script (e.g. python Models/LGBMRegressor_Model.py).  
4. Inspect saved outputs in Code/Results and Trained_Model.  
5. Validate predictions with python Load_Trained_Model_For_Prediction.py or python CNN_Prediction.py.

Notes:
- Many scripts currently use absolute paths — convert to relative paths or CLI/config variables for portability.
- MAPE implementations divide by y_true; avoid zero targets or add a small epsilon to prevent division-by-zero.

---
