# %%
import joblib 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#%%
dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Test_data/Combined_Simu_1_accel.csv')

X = dataset.drop(columns=['Time_Step', 'Force_Applied'], axis=1).values

Y = dataset.Force_Applied.values

#%% Data Standardization 
X_std = StandardScaler()
Y_std = StandardScaler()
X_Standardized = X_std.fit_transform(X)

Y_Reshaped = Y.reshape((Y.shape[0], 1))

Y_Actual_Standardized = Y_std.fit_transform(Y_Reshaped).reshape(-1)

#%% Load the model from the file 

SVR = joblib.load('/Users/mukul/Desktop/DLR_Internship/Code/Trained_Model/LGBMRegressor.pkl')

# Use the loaded model to make predictions for the entire test set
Y_Predicted_Standardized = SVR.predict(X_Standardized)

#%%# Importing the required Metrics from Sklearn
from sklearn.metrics import (mean_absolute_error, 
                            mean_squared_error, 
                            r2_score, 
                            explained_variance_score)

# Mean Absolute Error
MAE = mean_absolute_error(Y_Actual_Standardized, Y_Predicted_Standardized)
print(f'Mean Absolute Error: {MAE}')

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(Y_Actual_Standardized, Y_Predicted_Standardized)
print(f'Mean Absolute Percentage Error: {MAPE}%')

# Mean Squared Error
MSE = mean_squared_error(Y_Actual_Standardized, Y_Predicted_Standardized)
print(f'Mean Squared Error: {MSE}')

# Root Mean Squared Error
RMSE = mean_squared_error(Y_Actual_Standardized, Y_Predicted_Standardized, squared=False)
print(f'Root Mean Squared Error: {RMSE}')

# R-squared Error
r2 = r2_score(Y_Actual_Standardized, Y_Predicted_Standardized)
print(f'R-squared: {r2}')

# Explained Variance Score
EVS = explained_variance_score(Y_Actual_Standardized, Y_Predicted_Standardized)
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
model_name = 'Support_Vector_Regressor_Results'

# File path where you want to save the results
file_path = f'/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/Actual_Test_Data_Results/LGBMRegressor/{model_name}.txt'

# Writing the results to the text file
with open(file_path, 'w') as file:
    file.write(f"Results for model: {model_name}\n")
    file.write("=" * 40 + "\n")
    for key, value in results.items():
        file.write(f"{key}: {value}\n")

print(f"Results saved to {file_path}")

#%% Inverse transform the predicted values and actual values to the original scale

Actual_Values_In_Original_Scale = Y_std.inverse_transform(Y_Actual_Standardized.reshape(-1, 1)).flatten()

Predicted_Values_In_Original_Scale = Y_std.inverse_transform(Y_Predicted_Standardized.reshape(-1, 1)).flatten()

# #%% Plotting the scatter plot between actual values and predicted values

# # for 200 values in test data
# plt.figure(figsize=(10, 6))
# plt.scatter(range(50),Actual_Values_In_Original_Scale[:50,], label='Actual Force Applied', marker='o', s=100, c='c', edgecolors='k',linewidths=0.6)
# plt.scatter(range(50),Predicted_Values_In_Original_Scale[:50,], label='Force Predicted', marker='*', s=100, c='m', edgecolors='k',linewidths=0.6)
# plt.xlabel('Number of Samples')
# plt.ylabel('Force')
# plt.title('Actual vs Predicted Values of ,  Force Applied')
# plt.legend()
# #plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/Actual_Test_Data_Results/LGBMRegressor/LGBMRegressor.png')
# #plt.show()

# #%%
# # for total samples in test data
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(Actual_Values_In_Original_Scale)),Actual_Values_In_Original_Scale, label='Actual Force Applied', marker='o', s=100, c='c', edgecolors='k',linewidths=0.4)
# plt.scatter(range(len(Actual_Values_In_Original_Scale)),Predicted_Values_In_Original_Scale, label='Force Predicted', marker='*', s=100, c='m', edgecolors='g',linewidths=0.1)
# plt.xlabel('Number of Samples')
# plt.ylabel('Force')
# plt.title('Actual vs Predicted Values of Force Applied')
# plt.legend()
# #plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/Actual_Test_Data_Results/LGBMRegressor/LGBMRegressor1.png')
# #plt.show()

# %% Plot line Plot
plt.figure(figsize=(10, 6))
plt.plot(range(200), Actual_Values_In_Original_Scale[:200],label='Actual Force' )
plt.plot(range(200), Predicted_Values_In_Original_Scale[:200],label='Predicted Force')
plt.xlabel('Number of Samples')
plt.ylabel('Force')
plt.title('Line Plot Comparing actual & Predicted Force')
plt.legend()
plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/LGBMRegressor/Line_Plot_LGBMRegressor1.png')
plt.show()

#%%
# fig, ax1 = plt.subplots()
# plt.plot(range(200), Actual_Values_In_Original_Scale[:200],label='Actual Force',color='red' )
# #ax1.plot(range(10), Actual_Values_In_Original_Scale[:10], label='Actual Values', )
# ax1.set_xlabel('Number of Samples')
# ax1.set_ylabel('Actual Force', color='red')
# ax1.tick_params(axis='y', labelcolor='red')
# # Create a secondary y-axis to plot the predicted values
# ax2 = ax1.twinx()
# plt.plot(range(200), Predicted_Values_In_Original_Scale[:200],label='Predicted Force',color='orange')
# #ax2.plot(range(10), Predicted_Values_In_Original_Scale[:10], label='Predicted Values', color='red')
# ax2.set_ylabel('Predicted Force', color='orange')
# ax2.tick_params(axis='y', labelcolor='orange')
# plt.title('Line Plot Comparing Actual & Predicted Force')
# fig.tight_layout()  # Adjust layout to make room for both y-axes
# plt.show()

# %%
