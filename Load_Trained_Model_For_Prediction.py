# %%

from joblib import Parallel, delayed 
import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


#%%
dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Combine_Data.csv')

X = dataset.drop(columns=['Time_Step', 'Force_Applied'], axis=1)

Y = dataset.Force_Applied

#%% Data Standardization 
std = StandardScaler()

X_Train_Standardized = std.fit_transform(X)

Y_Reshaped = Y.to_numpy().reshape((Y.shape[0], 1))

Y_Train_Standardized = std.fit_transform(Y_Reshaped).reshape(-1)
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
  
# Use the loaded model to make predictions 
Y_prediction = SVR.predict(X_Test) 

# Mean Squared Error
MSE = mean_squared_error(Y_Test,Y_prediction)
print(f'Mean Squared Error: {MSE}')
# %%
