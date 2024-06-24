#%% Import Required Python Files
import Data_PreProcessing as DP

#%% Import the requied libraries
import pandas as pd 
import numpy as np  
import sktime
import sklearn
from sklearn.linear_model import SGDRegressor
import matplotlib as plt
from sklearn.metrics import mean_squared_error
import time

Start_Time = time.time()
print(f"Model training started.")
#%% Get the model to predict the focre applied

# Initialize the SVM regression model
Model = SGDRegressor()
# Train the model on the transformed training data

Model.fit(DP.X_Train_Transformed, DP.Y_Train)

# Make predictions on the test set

Y_Prediction = Model.predict(DP.X_Train_Transformed)

Current_Time = time.time()

print(f"Model training is done.\n It took {Current_Time-Start_Time} to train the model and make preditions.")
# Calculate the mean squared error
MSE = mean_squared_error(DP.Y_Train, Y_Prediction)
print(f'Mean Squared Error: {MSE}')

End_Time = time.time()

print(f"The process is completed and it took in total {End_Time-Start_Time} sec.")