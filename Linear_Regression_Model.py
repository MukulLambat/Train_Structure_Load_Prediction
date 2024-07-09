#%% Import Required Python Files
import Data_PreProcessing as DP

#%% Import the requied libraries
import pandas as pd 
import numpy as np  
import sktime
import sklearn
from sklearn import linear_model
import matplotlib as plt
import time

Start_Time = time.time()

print(f"Model training started.")
#%% Get the model to predict the focre applied

# Initialize the linear regression model
Model = linear_model.LinearRegression()
# Train the model on the transformed training data

Model.fit(DP.X_Train_Transformed, DP.Y_Train)

# Make predictions on the test set

Y_Prediction = Model.predict(DP.X_Train_Transformed)

Current_Time = time.time()

print(f" Model training is done.\n It took {Current_Time-Start_Time} to train the model and make preditions.")


# Importing the required Metrics from Sklearn
from sklearn.metrics import (mean_absolute_error, 
                            mean_squared_error, 
                            mean_squared_log_error, 
                            r2_score, 
                            explained_variance_score)


# Mean Absolute Error
MAE = mean_absolute_error(DP.Y_Train, Y_Prediction)
print(f'Mean Absolute Error: {MAE}')

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(DP.Y_Train, Y_Prediction)
print(f'Mean Absolute Percentage Error: {MAPE}%')

# Mean Squared Error
MSE = mean_squared_error(DP.Y_Train, Y_Prediction)
print(f'Mean Squared Error: {MSE}')

# Root Mean Squared Error
RMSE = mean_squared_error(DP.Y_Train, Y_Prediction, squared=False)
print(f'Root Mean Squared Error: {RMSE}')

# Mean Squared Log Error
MLSE = mean_squared_log_error(DP.Y_Train, Y_Prediction)
print(f'Mean Squared Logarithmic Error: {MLSE}')

# R-squared Error
r2 = r2_score(DP.Y_Train, Y_Prediction)
print(f'R-squared: {r2}')

# Explained Variance Score
EVS = explained_variance_score(DP.Y_Train, Y_Prediction)
print(f'Explained Variance Score: {EVS}')

# Collecting results to save them in the text file
results = {
    'Mean Absolute Error': MAE,
    'Mean Absolute Percentage Error': MAPE,
    'Mean Squared Error': MSE,
    'Root Mean Squared Error': RMSE,
    'Mean Squared Logarithmic Error': MLSE,
    'R-squared': r2,
    'Explained Variance Score': EVS
}

# Get model name
model_name = 'Linear_Regression_Model_Results'

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