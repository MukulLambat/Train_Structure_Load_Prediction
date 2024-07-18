# %%

from joblib import Parallel, delayed 
import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch.nn import Conv1d, MaxPool1d, Flatten, Linear, Module
from torch.nn.functional import relu


#%%
dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Combine_Data.csv')

X = dataset.drop(columns=['Time_Step', 'Force_Applied'], axis=1)

Y = dataset.Force_Applied

#%% Data Standardization 
X_std = StandardScaler()
Y_std = StandardScaler()
X_Train_Standardized = X_std.fit_transform(X)

Y_Reshaped = Y.to_numpy().reshape((Y.shape[0], 1))

Y_Train_Standardized = Y_std.fit_transform(Y_Reshaped).reshape(-1)
#%%
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train_Standardized,Y_Train_Standardized,
                                   shuffle = True,
                                   random_state=0, 
                                   train_size = 0.8,
                                   test_size= 0.2,
                                   stratify=None)

#%%
# Load the model from the file 
SVR = joblib.load('/Users/mukul/Desktop/DLR_Internship/Trained_Model/Support_Vector_Regressor.pkl') 
#%%

# Use the loaded model to make predictions for the entire test set
predicted_values_standardized = SVR.predict(X_Test)

#%%
# Inverse transform the predicted values and actual values to the original scale
predicted_values_original = Y_std.inverse_transform(predicted_values_standardized.reshape(-1, 1)).flatten()
actual_values_original = Y_std.inverse_transform(Y_Test.reshape(-1, 1)).flatten()

#%%
# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(actual_values_original, predicted_values_original, alpha=0.7)
#plt.plot([min(actual_values_original), max(actual_values_original)], [min(actual_values_original), max(actual_values_original)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.show()
