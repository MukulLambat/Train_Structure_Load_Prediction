#%% Import Required python files
import Actual_Data as AD

#%% Import Required Libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

Start_Time = time.time()

# %%# Check for missing values
Missing_Values = AD.Concatenated_Simulation_Result.isnull().sum().sum()
Concatenated_Simulation_Result = AD.Concatenated_Simulation_Result.dropna()

#%% Separate inputs (sensor data) and output (Force applied)

X = Concatenated_Simulation_Result.drop(columns=['Time_Step', 'Force_Applied'], axis=1)

Y = Concatenated_Simulation_Result.Force_Applied

#%% Statistics of the Data
Statistics = pd.DataFrame(X).describe()

#%% Data Standardization 
std = StandardScaler()

X_Train_Standardized = std.fit_transform(X)

Y_Reshaped = Y.to_numpy().reshape((Y.shape[0], 1))

Y_Train_Standardized = std.fit_transform(Y_Reshaped).reshape(-1)

#%% Split the Dataset using train test split function from sklearn

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train_Standardized, Y_Train_Standardized,
                                   shuffle = True,
                                   random_state=0, 
                                   train_size = 0.8,
                                   test_size= 0.2,
                                   stratify=None)

End_Time = time.time()

print(f"Data Pre-Processing is done and is completed in {End_Time-Start_Time} sec.")

