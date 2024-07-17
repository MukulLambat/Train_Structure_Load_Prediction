import time

#%% Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
# import 1D convolutional layer
from torch.nn import Conv1d

# import max pooling layer
from torch.nn import MaxPool1d

# import the flatten layer 
from torch.nn import Flatten

# import linear layer
from torch.nn import Linear

# import activation function (ReLU)
from torch.nn.functional import relu

# import libraries required for working with dataset from pytorch
from torch.utils.data import DataLoader, TensorDataset

# import SGD for optimizer
from torch.optim import SGD

# import Adam for optimizer
from torch.optim import Adam

# to measure the performance import MSELoss
from torch.nn import MSELoss

from sklearn.metrics import mean_squared_error
     
#%% read the csv file of final dataset
dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Combine_Data.csv')

# removed records with missing values
dataset = dataset.dropna()
# %%
X = dataset.drop(['Time_Step','Force_Applied'], axis=1).values
Y = dataset['Force_Applied'].values

# %% Split the Dataset using train test split function from sklearn
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,
                                   shuffle = True,
                                   random_state=0, 
                                   train_size = 0.8,
                                   test_size= 0.2,
                                   stratify=None)

# Standardize the dataset
scaler_X = StandardScaler()
X_Train = scaler_X.fit_transform(X_Train)
X_Test = scaler_X.transform(X_Test)

# Standardize the target
scaler_Y = StandardScaler()
Y_Train = scaler_Y.fit_transform(Y_Train.reshape(-1, 1)).flatten()
Y_Test = scaler_Y.transform(Y_Test.reshape(-1, 1)).flatten()

# %% defined model named as CnnRegressor and
# this model should be the subclass of torch.nn.Module 
class CNNRegressor(torch.nn.Module):
      
      # defined the initialization method
  def __init__(self, batch_size, inputs, outputs):
      
    # initialization of the superclass
    super(CNNRegressor, self).__init__()
    # store the parameters
    self.batch_size = batch_size
    self.inputs = inputs
    self.outputs = outputs
    
    # define the input layer
    self.input_layer = Conv1d(inputs, batch_size, 1, stride = 1)
    
    # define max pooling layer
    self.max_pooling_layer = MaxPool1d(1)
    
    # define other convolutional layers
    self.conv_layer1 = Conv1d(batch_size, 128, 1, stride = 3)
    self.conv_layer2 = Conv1d(128, 256, 1, stride = 3)
    self.conv_layer3 = Conv1d(256, 512, 1, stride = 3)

    # define the flatten layer
    self.flatten_layer = Flatten()

    # define the linear layer
    self.linear_layer = Linear(512, 128)

    # define the output layer
    self.output_layer = Linear(128, outputs)
    
    # define the method to feed the inputs to the model
  def feed(self, input):
    # input is reshaped to the 1D array and fed into the input layer
    input = input.reshape((self.batch_size, self.inputs, 1))

    # ReLU is applied on the output of input layer
    output = relu(self.input_layer(input))

    # max pooling is applied and then Convolutions are done with ReLU
    output = self.max_pooling_layer(output)
    output = relu(self.conv_layer1(output))

    output = self.max_pooling_layer(output)
    output = relu(self.conv_layer2(output))

    output = self.max_pooling_layer(output)
    output = relu(self.conv_layer3(output))

    # flatten layer is applied
    output = self.flatten_layer(output)

    # linear layer and ReLu is applied
    output = relu(self.linear_layer(output))

    # finally, output layer is applied
    output = self.output_layer(output)
    return output

#%% define the batch size  
batch_size = 128
model = CNNRegressor(batch_size, X.shape[1], 1)

# %%# define the method for calculating average MSE of given model
def model_loss(model, dataset, train = False, optimizer = None):
  # first calculated for the batches and at the end get the average
  performance = MSELoss()

  avg_loss = 0
  avg_score = 0
  count = 0

  for input, output in iter(dataset):
    # get predictions of the model for training set
    predictions = model.feed(input)

    # calculate loss of the model
    loss = performance(predictions, output)

    # compute the MSE score
    score = mean_squared_error(output.cpu().detach().numpy(), predictions.cpu().detach().numpy())

    if(train):
      # clear the errors
      optimizer.zero_grad()

      # compute the gradients for optimizer
      loss.backward()

      # use optimizer in order to update parameters
      # of the model based on gradients
      optimizer.step()

    # store the loss and update values
    avg_loss += loss.item()
    avg_score += score
    count += 1

  return avg_loss/count, avg_score/count
#%%# define the number of epochs
epochs = 300

# define the performance measure and optimizer
# optimizer = SGD( model.parameters(), lr= 1e-5)
optimizer = Adam(model.parameters(), lr = 0.001)

# to process with CPU, training set is converted into torch variable
inputs = torch.from_numpy(X_Train).float()
outputs = torch.from_numpy(Y_Train.reshape(Y_Train.shape[0],1)).float()

# create the DataLoader instance to work with batches
tensor = TensorDataset(inputs, outputs)
loader = DataLoader(tensor, batch_size, shuffle = True, drop_last = True)

# list to store loss values
loss_values = []
mse_values = []

# start timing
start_time = time.time()

# loop for number of epochs and calculate average loss
for epoch in range(epochs):
   # model is cycled through the batches
  avg_loss, avg_mse = model_loss(model, loader, train = True, optimizer = optimizer)
  loss_values.append(avg_loss)
  mse_values.append(avg_mse)
  print("Epoch " + str(epoch + 1) + ":\n\tLoss = " + str(avg_loss) + "\n\tMSE = " + str(avg_mse))

# end timing
end_time = time.time()
training_time = end_time - start_time

# print the total training time
print(f"Total training time: {training_time:.2f} seconds")

# plot the loss graph
plt.plot(range(epochs), loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# plot the MSE graph
plt.plot(range(epochs), mse_values, label='Training MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Training MSE Over Epochs')
plt.legend()
plt.show()
