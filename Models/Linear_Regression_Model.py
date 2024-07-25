# Import Required Python Files
import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to Process_Data
process_data_path = os.path.join(current_dir, '../Process_Data')

# Add Process_Data to the system path
sys.path.insert(0, process_data_path)

# Now you can import Data_PreProcessing
import Data_PreProcessing as DP

#%% Import the requied libraries
import pandas as pd 
import numpy as np  
import sktime
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import time
import seaborn as sns 
from joblib import Parallel, delayed 
import joblib 

Start_Time = time.time()

print(f"Model training started.")
#%% Get the model to predict the force applied

# Initialize the linear regression model
Model = linear_model.LinearRegression(fit_intercept=True, n_jobs = -1)
#%%
# Train the model on the transformed training data

Model.fit(DP.X_Train, DP.Y_Train)

# Save the model as a pickle in a file 
joblib.dump(Model, '/Users/mukul/Desktop/DLR_Internship/Code/Trained_Model/Linear_Regression.pkl') 

#%%
# # Make predictions on the train set

Y_Train_Prediction = Model.predict(DP.X_Train)

# Make predictions on the test set

Y_Test_Prediction = Model.predict(DP.X_Test)


Current_Time = time.time()

print(f" Model training is done.\n It took {Current_Time-Start_Time} to train the model and make preditions.")


# Importing the required Metrics from Sklearn
from sklearn.metrics import (mean_absolute_error, 
                            mean_squared_error, 
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
    'R-squared': r2,
    'Explained Variance Score': EVS
}

# Get model name
model_name = 'Linear_Regression_Results'

# File path where you want to save the results
file_path = f'/Users/mukul/Desktop/DLR_Internship/Code/Results/Validation_Results/Linear_Regression/{model_name}.txt'

# Writing the results to the text file
with open(file_path, 'w') as file:
    file.write(f"Results for model: {model_name}\n")
    file.write("=" * 40 + "\n")
    for key, value in results.items():
        file.write(f"{key}: {value}\n")

print(f"Results saved to {file_path}")

#%% Plotting the scatter plot between actual values and predicted values
#Inverse transform the predicted values and actual values to the original scale

Actual_Values_In_Original_Scale = DP.std.inverse_transform(DP.Y_Test.reshape(-1, 1)).flatten()

Predicted_Values_In_Original_Scale = DP.std.inverse_transform(Y_Test_Prediction.reshape(-1, 1)).flatten()

# for 200 values in test data
plt.figure(figsize=(10, 6))
plt.scatter(range(1600),Actual_Values_In_Original_Scale[:1600,], label='Actual Force Applied', marker='o', s=100, c='c', edgecolors='k',linewidths=0.6)
plt.scatter(range(1600),Predicted_Values_In_Original_Scale[:1600,], label='Force Predicted', marker='*', s=100, c='m', edgecolors='k',linewidths=0.6)
plt.xlabel('Number of Samples')
plt.ylabel('Force')
plt.title('Actual vs Predicted Values of Force Applied')
plt.legend()
plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Validation_Results/Linear_Regression/Linear_Regression.png')
#plt.show()

End_Time = time.time()

print(f"The process is completed and it took in total {End_Time-Start_Time} sec.")

# %%
