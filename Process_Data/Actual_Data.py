#%% Import the required Python file
import Reshape_CSVS

#%% Import Libraries
import pandas as pd
import numpy as np
import os
import glob
import re
import time

#%% 
Raw_Data_Path = '/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Data'
Start_Time = time.time() 

#%%
# Define a custom sorting key function
def extract_number(filename):
    # Use regular expression to extract the number from the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return float('inf')  # Return a large number if no number is found

#%% Get the List of aceeleration files in a list using glob
Acceleration_Files = glob.glob(os.path.join(Raw_Data_Path, 'Reshaped_Simu_*_accel_x.csv'))

# Sort the files based on the number present in the filename
Sorted_Acceleration_Files = sorted(Acceleration_Files, key=extract_number)

#print(Sorted_Acceleration_Files)

# Get the List of strain files in a list using glob
Strain_Files = glob.glob(os.path.join(Raw_Data_Path, 'Reshaped_Simu_*_export_strain.csv'))

# Sort the files based on the number present in the filename
Sorted_Strain_Files = sorted(Strain_Files, key=extract_number)

#print(Sorted_Strain_Files)

# %%
for index, file in enumerate(Sorted_Acceleration_Files):
    
    # Extract the file name from the path
    X_Acceleration_Name = os.path.basename(file)
    #print(X_Acceleration_Name)
    
    Split_File_Name = X_Acceleration_Name.split('_')
    # print(Split_File_Name)
    # print(index)
    
    # Read the x, y and z direction files and Force applied
    X_Acceleration = pd.read_csv(os.path.join(Raw_Data_Path, X_Acceleration_Name))
    Time_Step = X_Acceleration.iloc[:,0]
    X_Acceleration = X_Acceleration.iloc[:,1:]
    Y_Acceleration = pd.read_csv(os.path.join(Raw_Data_Path,f"{Split_File_Name[0]}_{Split_File_Name[1]}_{Split_File_Name[2]}_{Split_File_Name[3]}_y.csv")).iloc[:,1:]
    Z_Acceleration = pd.read_csv(os.path.join(Raw_Data_Path,(f"{Split_File_Name[0]}_{Split_File_Name[1]}_{Split_File_Name[2]}_{Split_File_Name[3]}_z.csv"))).iloc[:,1:]
    Force_Applied = pd.read_csv(os.path.join(Raw_Data_Path,f"Force_{Split_File_Name[2]}.csv"), header=None).iloc[:,1]
    Strain = pd.read_csv(os.path.join(Raw_Data_Path, Sorted_Strain_Files[index])).iloc[:,1:]
    
    # Empty Dataframe for Separate Dataset
    Dataset = pd.DataFrame()
    
    Dataset.insert(0,'Time_Step', Time_Step)
    
    # Iterate over each column index
    for i in range(len(X_Acceleration.columns)):
    # Extract columns from X, Y, and Z acceleration DataFrames and concatenate them horizontally
        X_Acceleration_Column = X_Acceleration.iloc[:, i]
        Y_Acceleration_Column = Y_Acceleration.iloc[:, i] 
        Z_Acceleration_Column = Z_Acceleration.iloc[:, i]
        
        # Concatenate the columns horizontally
        Temporary_Dataset = pd.concat([X_Acceleration_Column, Y_Acceleration_Column, Z_Acceleration_Column], axis=1)
        
        # Rename the columns to avoid duplicates, using the original column names
        Original_Column_Names = [X_Acceleration.columns[i], Y_Acceleration.columns[i], Z_Acceleration.columns[i]]        
        Temporary_Dataset.columns = [f'{Original_Column_Names[0]}', f'{Original_Column_Names[1]}', f'{Original_Column_Names[2]}']
        
        # Concatenate the temporary DataFrame with the Dataset DataFrame
        Dataset = pd.concat([Dataset, Temporary_Dataset], axis=1)
        
    # Concatenate the two DataFrames on axis 1 (columns)
    Dataset = pd.concat([Dataset, Strain], axis=1)
    
    # Check the number of rows in Dataset and Force Applied 
    Rows_Dataset = Dataset.shape[0]
    Rows_Force_Applied = Force_Applied.shape[0]

    # If the Force_Applied DataFrame has fewer rows, append the last row 
    if Rows_Force_Applied < Rows_Dataset:
        last_row = Force_Applied.iloc[[-1]].copy()
        while Force_Applied.shape[0] < Rows_Dataset:
            Force_Applied = pd.concat([Force_Applied, last_row], ignore_index=True)

    # Concatenate the two DataFrames on axis 1 (columns)
    Dataset = pd.concat([Dataset, Force_Applied], axis=1)

    # Rename the last column to "Force_Applied"
    Dataset.columns = list(Dataset.columns[:-1]) + ['Force_Applied']

    #Save the Dataset DataFrame to a CSV file
    Dataset.to_csv(f'/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Data/Combined_{Split_File_Name[1]}_{Split_File_Name[2]}_{Split_File_Name[3]}.csv', index=False)

# %% Now Concatnate each individual CSV to create the whole Dataset
Path_Combine_Individual_Simulation_Result = ('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Data')

# Get the List of aceeleration files in a list using glob
Individual_Simulation_Result = glob.glob(os.path.join(Path_Combine_Individual_Simulation_Result, 'Combined_Simu_*_accel.csv'))

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

Concatenated_Simulation_Result.to_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Combine_Data.csv', index=False)

#%%
End_Time = time.time()
 
print(f"Final Dataset created and is completed in {End_Time-Start_Time} sec.")

