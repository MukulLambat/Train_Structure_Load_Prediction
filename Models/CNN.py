import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, explained_variance_score)
import torch
from torch.nn import Conv1d, MaxPool1d, Flatten, Linear, MSELoss
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# Read the csv file of final dataset
dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Combine_Data.csv')

# Remove records with missing values
dataset = dataset.dropna()

# Prepare features and target
X = dataset.drop(['Time_Step', 'Force_Applied'], axis=1).values
Y = dataset['Force_Applied'].values

# Split the dataset
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, shuffle=True, random_state=0, train_size=0.8, test_size=0.2)

# Standardize the dataset
scaler_X = StandardScaler()
X_Train = scaler_X.fit_transform(X_Train)
X_Test = scaler_X.transform(X_Test)

# Standardize the target
scaler_Y = StandardScaler()
Y_Train = scaler_Y.fit_transform(Y_Train.reshape(-1, 1)).flatten()
Y_Test = scaler_Y.transform(Y_Test.reshape(-1, 1)).flatten()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model named as CNNRegressor
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
    
    def feed(self, input):
        batch_size = input.shape[0]  # Get the actual batch size
        input = input.reshape((batch_size, self.inputs, 1))
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

# Define the batch size  
batch_size = 128
model = CNNRegressor(batch_size, X.shape[1], 1).to(device)

# Define the method for calculating average MSE of given model
def model_loss(model, dataset, train=False, optimizer=None):
    performance = MSELoss()
    avg_loss = 0
    avg_score = 0
    count = 0

    for input, output in iter(dataset):
        input, output = input.to(device), output.to(device)
        predictions = model.feed(input)
        loss = performance(predictions, output)
        score = mean_squared_error(output.cpu().detach().numpy(), predictions.cpu().detach().numpy())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss += loss.item()
        avg_score += score
        count += 1

    return avg_loss / count, avg_score / count

# Define the number of epochs
epochs = 1

# Define the performance measure and optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Convert training set into torch variable
inputs = torch.from_numpy(X_Train).float().to(device)
outputs = torch.from_numpy(Y_Train.reshape(Y_Train.shape[0], 1)).float().to(device)

# Create the DataLoader instance to work with batches
tensor = TensorDataset(inputs, outputs)
loader = DataLoader(tensor, batch_size, shuffle=True, drop_last=True)

# List to store loss values
loss_values = []
mse_values = []

# Start timing
start_time = time.time()

# Loop for number of epochs and calculate average loss
for epoch in range(epochs):
    avg_loss, avg_mse = model_loss(model, loader, train=True, optimizer=optimizer)
    loss_values.append(avg_loss)
    mse_values.append(avg_mse)
    print("Epoch " + str(epoch + 1) + ":\n\tLoss = " + str(avg_loss) + "\n\tMSE = " + str(avg_mse))

# End timing
end_time = time.time()
training_time = end_time - start_time

# Print the total training time
print(f"Total training time: {training_time:.2f} seconds")

# Plot the loss graph
plt.plot(range(epochs), loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Validation_Results/CNN/CNN_training_loss_plot.png')
plt.clf()

# Save the trained model
torch.save(model, "/Users/mukul/Desktop/DLR_Internship/Code/Trained_Model/CNN_Model.pth")

# Define the prediction function
def predict(model, loader):
    model.eval()  # set the model to evaluation mode
    predictions = []
    actuals = []

    with torch.no_grad():
        for input, output in iter(loader):
            input = input.to(device)
            output = output.to(device)
            preds = model.feed(input)
            predictions.append(preds.cpu().numpy())
            actuals.append(output.cpu().numpy())

    return np.vstack(predictions).flatten(), np.vstack(actuals).flatten()

# Prepare the test dataset
inputs_test = torch.from_numpy(X_Test).float().to(device)
outputs_test = torch.from_numpy(Y_Test.reshape(Y_Test.shape[0], 1)).float().to(device)
tensor_test = TensorDataset(inputs_test, outputs_test)
loader_test = DataLoader(tensor_test, batch_size, shuffle=False, drop_last=False)

# Make predictions on the test set
Y_Test_Prediction, Y_Test_Actual = predict(model, loader_test)

# Calculate evaluation metrics
MAE = mean_absolute_error(Y_Test_Actual, Y_Test_Prediction)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(Y_Test_Actual, Y_Test_Prediction)
MSE = mean_squared_error(Y_Test_Actual, Y_Test_Prediction)
RMSE = mean_squared_error(Y_Test_Actual, Y_Test_Prediction, squared=False)
r2 = r2_score(Y_Test_Actual, Y_Test_Prediction)
EVS = explained_variance_score(Y_Test_Actual, Y_Test_Prediction)

# Print the metrics
print(f'Mean Absolute Error: {MAE}')
print(f'Mean Absolute Percentage Error: {MAPE}%')
print(f'Mean Squared Error: {MSE}')
print(f'Root Mean Squared Error: {RMSE}')
print(f'R-squared: {r2}')
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
model_name = 'CNN_Regressor'

# File path where you want to save the results
file_path = f'/Users/mukul/Desktop/DLR_Internship/Code/Results/Validation_Results/CNN/{model_name}.txt'

# Writing the results to the text file
with open(file_path, 'w') as file:
    file.write(f"Results for model: {model_name}\n")
    file.write("=" * 40 + "\n")
    for key, value in results.items():
        file.write(f"{key}: {value}\n")


#%% Plotting the scatter plot between actual values and predicted values
#Inverse transform the predicted values and actual values to the original scale

Actual_Values_In_Original_Scale = scaler_Y.inverse_transform(Y_Test.reshape(-1, 1)).flatten()

Predicted_Values_In_Original_Scale = scaler_Y.inverse_transform(Y_Test_Prediction.reshape(-1, 1)).flatten()

# for 200 values in test data
plt.figure(figsize=(10, 6))
plt.scatter(range(1600),Actual_Values_In_Original_Scale[:1600,], label='Actual Force Applied', marker='o', s=100, c='c', edgecolors='k',linewidths=0.6)
plt.scatter(range(1600),Predicted_Values_In_Original_Scale[:1600,], label='Force Predicted', marker='*', s=100, c='m', edgecolors='k',linewidths=0.6)
plt.xlabel('Number of Samples')
plt.ylabel('Force')
plt.title('Actual vs Predicted Values of Force Applied')
plt.legend()
plt.savefig('/Users/mukul/Desktop/DLR_Internship/Code/Results/Validation_Results/CNN/CNN123.png')
#plt.show()
plt.close()
