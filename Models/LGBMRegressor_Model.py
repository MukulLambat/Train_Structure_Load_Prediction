#%%Add the directory containing Data_PreProcessing.py to the system path
import sys
sys.path.append('/Users/mukul/Desktop/DLR_Internship/Code/Process_Data')

# Import Required Python Files
import Data_PreProcessing as DP

#%% Import the required libraries
import pandas as pd 
import numpy as np  
import sktime
import sklearn
import lightgbm 
from lightgbm import LGBMRegressor
import matplotlib as plt
import time
from joblib import Parallel, delayed 
import joblib 

Start_Time = time.time()
print(f"Model training started.")
#%% Get the model to predict the focre applied

# Initialize the regression model
Model = LGBMRegressor(boosting_type='gbdt',
                      num_leaves = 60,
                      n_estimators = 500,
                      learning_rate = 0.01, 
                      class_weight = 'balanced', 
                      random_state = 112, 
                      n_jobs = -1)
# Train the model on the transformed training data

Model.fit(DP.X_Train, DP.Y_Train)

# Save the model as a pickle in a file 
joblib.dump(Model, '/Users/mukul/Desktop/DLR_Internship/Trained_Model/LGBMRegressor.pkl') 

# Make predictions on the Train set

Y_Train_Prediction = Model.predict(DP.X_Train)

# Make predictions on the Test set

Y_Test_Prediction = Model.predict(DP.X_Test)

Current_Time = time.time()

print(f"Model training is done.\n It took {Current_Time-Start_Time} to train the model and make preditions.")

# Imorting the required metrics from sklearn

from sklearn.metrics import (mean_absolute_error, 
                            mean_squared_error, 
                            mean_squared_log_error, 
                            r2_score, 
                            explained_variance_score)


# Mean Absolute Error
MAE = mean_absolute_error(DP.Y_Test, Y_Test_Prediction)
print(f'Mean Absolute Error: {MAE}')

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(DP.Y_Test, Y_Test_Prediction)
print(f'Mean Absolute Percentage Error: {MAPE}%')

# Mean Squared Error
MSE = mean_squared_error(DP.Y_Test, Y_Test_Prediction)
print(f'Mean Squared Error: {MSE}')

# Root Mean Squared Error
RMSE = mean_squared_error(DP.Y_Test, Y_Test_Prediction, squared=False)
print(f'Root Mean Squared Error: {RMSE}')

# # Mean Squared Log Error
# MLSE = mean_squared_log_error(DP.Y_Test, Y_Test_Prediction)
# print(f'Mean Squared Logarithmic Error: {MLSE}')

# R-squared Error
r2 = r2_score(DP.Y_Test, Y_Test_Prediction)
print(f'R-squared: {r2}')

# Explained Variance Score
EVS = explained_variance_score(DP.Y_Test, Y_Test_Prediction)
print(f'Explained Variance Score: {EVS}')

# Collecting results to save them in the text file
results = {
    'Mean Absolute Error': MAE,
    'Mean Absolute Percentage Error': MAPE,
    'Mean Squared Error': MSE,
    'Root Mean Squared Error': RMSE,
    #'Mean Squared Logarithmic Error': MLSE,
    'R-squared': r2,
    'Explained Variance Score': EVS
}

# Get model name
model_name = 'LGBMRegressor_Model_Results'

# File path where you want to save the results
file_path = f'/Users/mukul/Desktop/DLR_Internship/Results/{model_name}.txt'

# Writing the results to the text file
with open(file_path, 'w') as file:
    file.write(f"Results for model: {model_name}\n")
    file.write("=" * 40 + "\n")
    for key, value in results.items():
        file.write(f"{key}: {value}\n")

print(f"Results saved to {file_path}")

End_Time = time.time()

print(f"The process is completed and it took in total {End_Time-Start_Time} sec.")