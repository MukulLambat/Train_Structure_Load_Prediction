#%% Import Requird python files
#import Actual_Data as AD

#%% Import Required Libraries
import pandas as pd 
import numpy as np
import sklearn
import numba
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.preprocessing import StandardScaler

Start_Time = time.time()

# %%# Check for missing values
#Missing_Values = AD.Concatenated_Simulation_Result.isnull().sum().sum()
# Print the count of missing values per column
# print("\nTotal count of missing values:")
# print(Missing_Values)

#print(Dataset.head(2))
#%% Separate inputs (sensor data) and output (Force applied)
Dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Combined_Data/Combined_Simu_1_accel.csv')

X = Dataset.drop(columns=['Time_Step', 'Force_Applied']).values
Y = Dataset['Force_Applied'].values

#%% Split the Dataset using train test split function from sklearn

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,
                                   shuffle = True,
                                   random_state=0, 
                                   train_size = 0.8,
                                   test_size= 0.2,
                                   stratify=None)

# print("X_X_Train:",X_Train[0:2,1:3])
# print("Y_Train:",Y_Train[1:3])
# print("X_Test:",X_Test[0:2,1:3])
# print("Y_Test:",Y_Test[1:3])

#%% Data Standardization 
std = StandardScaler()

X_Train_Standardized = std.fit_transform(X_Train)

X_Test_Standardized = std.fit_transform(X_Test)

#print("X_Train_Norm:",X_Train_Standardized[0:2,1:3])

#%% Data Normalization 

# fit scaler on training data
Normalizer_Model = MinMaxScaler().fit(X_Train_Standardized)

# transform training data
X_Train_Normalized = Normalizer_Model.transform(X_Train_Standardized)

# transform testing data
X_Test_Normalized = Normalizer_Model.transform(X_Test_Standardized)
#print("X_Train_Norm:",X_Train_Normalized[0:2,1:3])
# print("X_Test-Norm:",X_Test_Normalized[0:2,1:3])

#%% Reshape the data for miniROCKET
# miniROCKET expects data in the format (samples, timesteps, features)
# Here each sample is a single time step, so timesteps = 1

# X_Train_Reshaped = X_Train_Normalized.reshape((X_Train_Normalized.shape[0], 1, X_Train_Normalized.shape[1]))
# X_Test_Reshaped = X_Test_Normalized.reshape((X_Test_Normalized.shape[0], 1, X_Test_Normalized.shape[1]))

# Print the shapes of the prepared data
# print("X_train shape:", X_Train_Reshaped.shape)
# print("y_train shape:", Y_Train.shape)
# print("X_test shape:", X_Test_Reshaped.shape)
# print("y_test shape:", Y_Test.shape)

# # Print the reshaped data to see the format
# print("X_train reshaped:\n", X_Train_Reshaped)
# print("X_test reshaped:\n", X_Test_Reshaped)

#%% Transformation of input multivariate time series using  MiniRocket-Multivariate

# Minirocket_Model = MiniRocketMultivariate(n_jobs=-1, random_state=9)
# Minirocket_Model.fit(X_Train_Reshaped, Y_Train)
# X_Train_Transformed = Minirocket_Model.transform(X_Train_Reshaped)
# X_Test_Transformed = Minirocket_Model.transform(X_Test_Reshaped)
# # print(f'The transformed train dataset has shape of {X_train_Transformed.shape}, \n '
# #       f'and the transformed test dataset has shape {X_test_Transformed.shape}')

End_Time = time.time()

print(f"Data Pre-Processing is done and is completed in {End_Time-Start_Time} sec.")
