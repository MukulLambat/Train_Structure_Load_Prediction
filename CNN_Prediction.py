#%%

from joblib import Parallel, delayed 
import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch.nn import Conv1d, MaxPool1d, Flatten, Linear, Module
from torch.nn.functional import relu
import numpy as np

#%%
# Define your CNN model
class CNNRegressor(torch.nn.Module):
    def __init__(self, batch_size, inputs, outputs):
        super(CNNRegressor, self).__init__()
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        self.input_layer = Conv1d(inputs, batch_size, 1, stride=1)
        self.max_pooling_layer = MaxPool1d(1)
        self.conv_layer1 = Conv1d(batch_size, 128, 1, stride=3)
        self.conv_layer2 = Conv1d(128, 256, 1, stride=3)
        self.conv_layer3 = Conv1d(256, 512, 1, stride=3)
        self.flatten_layer = Flatten()
        self.linear_layer = Linear(512, 128)
        self.output_layer = Linear(128, outputs)
    
    def forward(self, input):
        input = input.reshape((self.batch_size, self.inputs, 1))
        output = relu(self.input_layer(input))
        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer1(output))
        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer2(output))
        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer3(output))
        output = self.flatten_layer(output)
        output = relu(self.linear_layer(output))
        output = self.output_layer(output)
        return output

#%%
dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Test_data/Combined_Simu_1_accel.csv')

X = dataset.drop(columns=['Time_Step', 'Force_Applied'], axis=1)

Y = dataset.Force_Applied

#%% Data Standardization 
X_std = StandardScaler()
Y_std = StandardScaler()
X_Train_Standardized = X_std.fit_transform(X)

Y_Reshaped = Y.to_numpy().reshape((Y.shape[0], 1))

Y_Train_Standardized = Y_std.fit_transform(Y_Reshaped).reshape(-1)

#%%
# Load the trained model
model = torch.load('/Users/mukul/Desktop/DLR_Internship/Code/Trained_Model/CNN_Model.pth')
model.eval()

# %%
# Convert test data to torch tensors
X_Test_Tensor = torch.tensor(X_Train_Standardized, dtype=torch.float32)
Y_Test_Tensor = torch.tensor(Y_Train_Standardized, dtype=torch.float32).unsqueeze(1)

#%%
# Make predictions on the test data
model.batch_size = X_Test_Tensor.shape[0]  # Set the batch size to the test size
with torch.no_grad():
    Y_Pred_Tensor = model(X_Test_Tensor)
    Y_Pred = Y_Pred_Tensor.numpy().flatten()

# Inverse transform the standardized predictions and actual values
Y_Pred_Original = Y_std.inverse_transform(Y_Pred.reshape(-1, 1)).flatten()
Y_Test_Original = Y_std.inverse_transform(Y_Train_Standardized.reshape(-1, 1)).flatten()

# %%
# Function to calculate Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate evaluation metrics
MAE = mean_absolute_error(Y_Train_Standardized, Y_Pred)
MAPE = mean_absolute_percentage_error(Y_Train_Standardized, Y_Pred)
MSE = mean_squared_error(Y_Train_Standardized, Y_Pred)
RMSE = mean_squared_error(Y_Train_Standardized, Y_Pred, squared=False)
r2 = r2_score(Y_Test_Original, Y_Pred)

# Print the metrics
print(f'Mean Absolute Error: {MAE}')
print(f'Mean Absolute Percentage Error: {MAPE}%')
print(f'Mean Squared Error: {MSE}')
print(f'Root Mean Squared Error: {RMSE}')
print(f'R-squared: {r2}')

# Collecting results to save them in the text file
results = {
    'Mean Absolute Error': MAE,
    'Mean Absolute Percentage Error': MAPE,
    'Mean Squared Error': MSE,
    'Root Mean Squared Error': RMSE,
    'R-squared': r2,
}

# Get model name
model_name = 'CNN_Regressor'

# File path where you want to save the results
file_path = f'/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/CNN/{model_name}.txt'

# Save the results to the file
with open(file_path, 'w') as file:
    file.write(f"Results for model: {model_name}\n")
    file.write("=" * 40 + "\n")
    for key, value in results.items():
        file.write(f'{key}: {value}\n')
        
#%%# for 200 values in test data
plt.figure(figsize=(10, 6))
plt.scatter(range(50),Y_Test_Original[:50,], label='Actual Force Applied', marker='o', s=100, c='c', edgecolors='k',linewidths=0.6)
plt.scatter(range(50),Y_Pred_Original[:50,], label='Force Predicted', marker='*', s=100, c='m', edgecolors='k',linewidths=0.6)
plt.xlabel('Number of Samples')
plt.ylabel('Force')
plt.title('Actual vs Predicted Values of Force Applied')
plt.legend()
plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/Actual_Test_Data_Results/CNN/CNN.png')
#plt.show()

#%%
# for total samples in test data
plt.figure(figsize=(10, 6))
plt.scatter(range(len(Y_Test_Original)),Y_Test_Original, label='Actual Force Applied', marker='o', s=100, c='c', edgecolors='k',linewidths=0.4)
plt.scatter(range(len(Y_Test_Original)),Y_Pred_Original, label='Force Predicted', marker='*', s=100, c='m', edgecolors='g',linewidths=0.1)
plt.xlabel('Number of Samples')
plt.ylabel('Force')
plt.title('Actual vs Predicted Values of Force Applied')
plt.legend()
plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/CNN/CNN1.png')
#plt.show()

# %% Plot line Plot
plt.figure(figsize=(10, 6))
plt.plot(range(200), Y_Test_Original[:200],label='Actual Force' )
plt.plot(range(200), Y_Pred_Original[:200],label='Predicted Force')
plt.xlabel('Number of Samples')
plt.ylabel('Force')
plt.title('Line Plot Comparing actual & Predicted Force')
plt.legend()
plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Test_Data/CNN/Line_Plot_CNN1.png')
plt.show()
# %%
