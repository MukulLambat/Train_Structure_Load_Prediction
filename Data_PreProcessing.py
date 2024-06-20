#%% Import Requird python files
import New_Dataset

#%% Import Required Libraries
import pandas as pd 
import numpy as np
import sklearn
import numba
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocketMultivariate
 
#%% Read the dataset
Dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Data/Combine_Data.csv')
# %%# Check for missing values
Missing_Values = Dataset.isnull().sum().sum()
# Print the count of missing values per column
print("\nTotal count of missing values:")
print(Missing_Values)

#print(Dataset.head(2))
#%% Separate inputs (sensor data) and output (Force applied)

X = Dataset.drop(columns=['Time_Step', 'Applied_Force']).values
Y = Dataset['Applied_Force'].values

#%% Split the Dataset using train test split function from sklearn

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,
                                   shuffle = True,
                                   random_state=0, 
                                   train_size = 0.8,
                                   test_size= 0.2,
                                   stratify=None)

#%% Reshape the data for miniROCKET
# miniROCKET expects data in the format (samples, timesteps, features)
# Here each sample is a single time step, so timesteps = 1
X_Train_Reshaped = X_Train.reshape((X_Train.shape[0], 1, X_Train.shape[1]))
X_Test_Reshaped = X_Test.reshape((X_Test.shape[0], 1, X_Test.shape[1]))

# Print the shapes of the prepared data
print("X_train shape:", X_Train_Reshaped.shape)
print("y_train shape:", Y_Train.shape)
print("X_test shape:", X_Test_Reshaped.shape)
print("y_test shape:", Y_Test.shape)

# Print the reshaped data to see the format
print("X_train reshaped:\n", X_Train_Reshaped)
print("X_test reshaped:\n", X_Test_Reshaped)


# %%
# Transformation of input multivariate time series using  MiniRocket-Multivariate
minirocket = MiniRocketMultivariate(n_jobs=-1, random_state=9)
minirocket.fit(X_Train_Reshaped, Y_Train)
X_train_Transformed = minirocket.transform(X_Train_Reshaped)
X_test_Transformed = minirocket.transform(X_Test_Reshaped)
print(f'The transformed train dataset has shape of {X_train_Transformed.shape}, \n '
      f'and the transformed test dataset has shape {X_test_Transformed.shape}')

