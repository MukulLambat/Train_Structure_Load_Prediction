''' This Scripts combines the simulation results obtained and create the dataset.'''

#%% Import Libraries
import pandas as pd
import numpy as np
import os
import glob
import re
#%%
Raw_Data_path = '/Users/mukul/Desktop/DLR_Internship/Data/Raw_Data'
 
# Define a custom sorting key function
def extract_number(filename):
    # Use regular expression to extract the number from the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return float('inf')  # Return a large number if no number is found

# Get the List of aceeleration files in a list using glob
Acceleration_Files = glob.glob(os.path.join(Raw_Data_path, 'Load_*_*Xa.csv'))

# Sort the files based on the number present in the filename
Sorted_Acceleration_Files = sorted(Acceleration_Files, key=extract_number)

for file in Sorted_Acceleration_Files:

    # Extract the file name from the path
    X_Acceleration_Name = os.path.basename(file)

    Split_File_Name = X_Acceleration_Name.split('_')

    # Read the x, y and z direction files and Force applied
    X_Acceleration = pd.read_csv(os.path.join(Raw_Data_path, X_Acceleration_Name))
    Y_Acceleration = pd.read_csv(os.path.join(Raw_Data_path,f"{Split_File_Name[0]}_{Split_File_Name[1]}_Ya.csv"))
    Z_Acceleration = pd.read_csv(os.path.join(Raw_Data_path,(f"{Split_File_Name[0]}_{Split_File_Name[1]}_Za.csv")))
    Force_Applied = pd.read_csv(os.path.join(Raw_Data_path,f"{Split_File_Name[0]}_{Split_File_Name[1]}_curve.csv"))
    
    # Empty Dataframe for Separate Dataset
    Dataset = pd.DataFrame()
    
    # Iterate over each column index
    for i in range(len(X_Acceleration.columns)):
    # Extract columns from X, Y, and Z acceleration DataFrames and concatenate them horizontally
        X_Acceleration_Column = X_Acceleration.iloc[:, i]
        Y_Acceleration_Column = Y_Acceleration.iloc[:, i] 
        Z_Acceleration_Column = Z_Acceleration.iloc[:, i]
        
        # Concatenate the columns horizontally
        Temporary_Dataset = pd.concat([X_Acceleration_Column, Y_Acceleration_Column, Z_Acceleration_Column], axis=1)
        
        # Rename the columns to avoid duplicates, using the original column names
        original_colnames = [X_Acceleration.columns[i], Y_Acceleration.columns[i], Z_Acceleration.columns[i]]        
        Temporary_Dataset.columns = [f'X_Acc_{i}_{original_colnames[0]}', f'Y_Acc_{i}_{original_colnames[1]}', f'Z_Acc_{i}{original_colnames[2]}']
        
        # Concatenate the temporary DataFrame with the Dataset DataFrame
        Dataset = pd.concat([Dataset, Temporary_Dataset], axis=1)
    
    # Extract the columns from Force Applied
    Time_Step = Force_Applied.iloc[:, 0]
    Applied_Force_Values = Force_Applied.iloc[:, 1]
    
    
    # Insert Time_Step at the first position
    Dataset.insert(0, 'Time_Step', Time_Step)

    # Append Applied_Force_Values at the last position
    Dataset['Applied_Force'] = Applied_Force_Values    
    
    # Save the Dataset DataFrame to a CSV file
    Dataset.to_csv(f'/Users/mukul/Desktop/DLR_Internship/Data/Combined_Load_&_Force/Combined_{Split_File_Name[0]}_{Split_File_Name[1]}_Force.csv', index=False)

# %% Now Concatnate each individual CSV to create the whole Dataset
Path_Combine_Individual_Simulation_Result = ('/Users/mukul/Desktop/DLR_Internship/Data/Combined_Load_&_Force')

# Get the List of aceeleration files in a list using glob
Individual_Simulation_Result = glob.glob(os.path.join(Path_Combine_Individual_Simulation_Result, 'Combined_Load_*_Force.csv'))

# Sort the files based on the number present in the filename
Sorted_Simulation_Result = sorted(Individual_Simulation_Result, key=extract_number)

# Empty list to store Individual Dataframe
Simulation_Result = []

for Each_Simulation in Sorted_Simulation_Result:
    df = pd.read_csv(Each_Simulation)
    Simulation_Result.append(df)
    
# Concatenate all DataFrames along axis 0 (vertically)
Concatenated_Simulation_Result = pd.concat(Simulation_Result, axis=0, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file

Concatenated_Simulation_Result.to_csv('/Users/mukul/Desktop/DLR_Internship/Data/Combine_Data.csv', index=False)
