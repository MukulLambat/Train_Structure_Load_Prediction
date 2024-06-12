'''This python script combines the data from various acceleration and force CSV files
into a single CSV file for model training.'''

# %% Importing required Libraries
import pandas as pd
import numpy as np
import glob
import os
import re
# %% File path for the Working Directory

Working_Directory =  '/Users/mukul/Desktop/DLR_Internship/code'

# %% Path of the directories for loading and saving files

Acceleration_Data_Path = '/Users/mukul/Desktop/DLR_Internship/Data/Load'
Applied_Force_Path = '/Users/mukul/Desktop/DLR_Internship/Data/The_Force'
Each_Combined_Load_And_Forces = '/Users/mukul/Desktop/DLR_Internship/Data/Each_Combined_Load_&_Force'

#%% Iterate through the Accelereration and Force data to combine force data to each acceleration data

# Get list of Load files using glob pattern

Acceleration_Data = glob.glob(os.path.join(Acceleration_Data_Path, 'Load_*_*a.csv'))


# Iterate through each Load (acceleration) file
for Each_Acceleration_File in Acceleration_Data:
    
    # Extract the file name from the path
    Each_Acceleration_File_Name = os.path.basename(Each_Acceleration_File)
    
    # Construct the corresponding Force file name
    Part_File_Name_1 = Each_Acceleration_File_Name.split('.') # This is to provide proper name save combined csv file
    Part_File_Name = Each_Acceleration_File_Name.split('_') # This is to prepare Force File name
    Splitted_Name = f"{Part_File_Name[0]}_{Part_File_Name[1]}"
    Force_File = f"{Splitted_Name}_curve.csv"

    # Construct full paths to the Load and Force files
    
    Each_Acceleration_Data_File_Path = os.path.join(Acceleration_Data_Path, Each_Acceleration_File_Name)
    Corresponding_Force_File_Path = os.path.join(Applied_Force_Path, Force_File)

    # Read the Load and Force CSV files
    if os.path.exists(Corresponding_Force_File_Path):
        Load_df = pd.read_csv(Each_Acceleration_Data_File_Path)
        Force_df = pd.read_csv(Corresponding_Force_File_Path)

        # Combine the data as we want to concatenate columns side by side
        Combined_df = pd.concat([Load_df, Force_df], axis=1)

        # Save the combined dataframe to the output folder
        output_file_name = f"Combined_{Part_File_Name_1[0]}_Force.csv"
        output_file_path = os.path.join(Each_Combined_Load_And_Forces, output_file_name)
        Combined_df.to_csv(output_file_path, index=False)
    else:
        print(f"Force file {Force_File} does not exist. Skipping.")

#%% Combine all the individual CSV files into single Dataset

All_Files = os.listdir(Each_Combined_Load_And_Forces)

# List to hold individual DataFrames
Dataset = []

# Iterate through all the files in directory
for Each_File in All_Files:
    Each_File_Path = os.path.join(Each_Combined_Load_And_Forces, Each_File)
    if Each_File.endswith('.csv'):
        df = pd.read_csv(Each_File_Path)
        Dataset.append(df)
        
# Concatenate all DataFrames in the list into one DataFrame
Combined_df = pd.concat(Dataset, ignore_index=True)

# Save the combined DataFrame to a new CSV file
Combined_df.to_csv('/Users/mukul/Desktop/DLR_Internship/Data/Dataset.csv', index=False)

#%% Time step columns is moved to first the postion
Dataset = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Data/Dataset.csv')

# Extract the 9th column
column_9 = Dataset.iloc[:, 9]  

# Delete the 9 th column from its original position
df = Dataset.drop(Dataset.columns[9], axis=1)

# Insert the extracted column at the first position
df.insert(0, 'Time Step', column_9)

# Write the modified DataFrame back to a CSV file
df.to_csv('/Users/mukul/Desktop/DLR_Internship/Data/Dataset_Modified.csv', index=False)    