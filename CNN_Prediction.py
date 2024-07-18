#%%
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
# Load the trained model
model = torch.load('/Users/mukul/Desktop/DLR_Internship/Code/Trained_Model/CNN_Model.pth')
model.eval()

# %%
# Convert test data to torch tensors
X_Test_Tensor = torch.tensor(X_Test, dtype=torch.float32)
Y_Test_Tensor = torch.tensor(Y_Test, dtype=torch.float32).unsqueeze(1)

#%%
# Make predictions on the test data
model.batch_size = X_Test_Tensor.shape[0]  # Set the batch size to the test size
with torch.no_grad():
    Y_Pred_Tensor = model(X_Test_Tensor)
    Y_Pred = Y_Pred_Tensor.numpy().flatten()

# Inverse transform the standardized predictions and actual values
Y_Pred_Original = Y_std.inverse_transform(Y_Pred.reshape(-1, 1)).flatten()
Y_Test_Original = Y_std.inverse_transform(Y_Test.reshape(-1, 1)).flatten()

#%%
# Plot the scatter plot between actual and predicted values
plt.scatter(Y_Test_Original, Y_Pred_Original)
plt.xlabel('Actual Force Applied')
plt.ylabel('Predicted Force Applied')
plt.title('Actual vs Predicted Force Applied')
plt.show()
# %%
