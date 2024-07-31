#%% import pandas as pd
import pandas as pd
import numpy as np
import os
import glob
import re
import os
import pandas as pd


#%% 
Raw_Data_Path = '/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Data'

#%% Get the List of aceeleration files in a list using glob
Acceleration_Files = glob.glob(os.path.join(Raw_Data_Path, 'Reshaped_Simu_*_accel_*.csv' ))

#Get the List of aceeleration files in a list using glob

Strain_Files = glob.glob(os.path.join(Raw_Data_Path, 'Reshaped_Simu_*_export_strain.csv' ))

# Combine two list of files 
All_Files = Acceleration_Files + Strain_Files

#%% The function to Process the Reshaped files and create mean of them row wise

def process_csv_files(file_paths):
    for file_path in file_paths:
                
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Select all columns except the first one
        X = df.iloc[:,1:]
        
        # Get the number of columns in X
        num_of_columns = X.shape[1]
        
        # Determine the split point for columns
        columns = int(num_of_columns/2)
        
        # Compute the row-wise mean of the first half of the columns
        mean_first_four = X.iloc[:, :columns].mean(axis=1)
        
        # Compute the row-wise mean of the second half of the columns
        mean_last_four = X.iloc[:, (columns+1):].mean(axis=1)
        
        # Create a new DataFrame with the means
        result_df = pd.DataFrame({
            X.columns[0]: mean_first_four,
            X.columns[1]: mean_last_four
        })
        # Construct the new file name
        base_name = os.path.basename(file_path)
        
        # Remove the file extension i.e. name without .csv
        base_name_Without_Extension = os.path.splitext(base_name)[0]
        
        base_name_split = base_name.split('_')
        
        new_file_name = f"Mean_{base_name_split[1]}_{base_name_split[2]}_{base_name_split[3]}_{base_name_split[4]}"
                
        # Save the new DataFrame to a new CSV file
        result_df.to_csv(f'/Users/mukul/Desktop/DLR_Internship/Actual_Data/check_data/{new_file_name}', index=False)

process_csv_files(All_Files)

#%% 
Raw_Data_Path = '/Users/mukul/Desktop/DLR_Internship/Actual_Data/One'

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
Force_Files = glob.glob(os.path.join(Raw_Data_Path, 'Force_*.csv'))

# Sort the files based on the number present in the filename
Sorted_Force_Files = sorted(Force_Files, key=extract_number)

#print(Sorted_Force_Files)
#%%
for file in Sorted_Force_Files:
    
    # Extract the file name from the path
    Force_File_Name = os.path.basename(file)
    #print(X_Acceleration_Name)
    
    # Remove the file extension i.e. name without .csv
    base_name_Without_Extension = os.path.splitext(Force_File_Name)[0]
    
    Split_File_Name = base_name_Without_Extension.split('_')
    # print(Split_File_Name)
    # print(index)
    
    X_Acceleration = pd.read_csv(os.path.join(Raw_Data_Path, f'Mean_Simu_{Split_File_Name[1]}_accel_x.csv'), header=0)
    Y_Acceleration = pd.read_csv(os.path.join(Raw_Data_Path,f'Mean_Simu_{Split_File_Name[1]}_accel_y.csv'), header=0)
    Z_Acceleration = pd.read_csv(os.path.join(Raw_Data_Path,f'Mean_Simu_{Split_File_Name[1]}_accel_z.csv'),header=0)
    Strain = pd.read_csv(os.path.join(Raw_Data_Path, f'Mean_Simu_{Split_File_Name[1]}_export_strain.csv'), header=0)
    
    # Read the x, y and z direction files and Force applied
    Force_Applied = pd.read_csv(os.path.join(Raw_Data_Path, Force_File_Name),header=None,names=['Time_Step', 'Force_Applied'])
    Time_Step = Force_Applied['Time_Step']
    Force_Applied = Force_Applied['Force_Applied']
    
    # If the Force_Applied DataFrame has fewer rows, append the last row 
    if Force_Applied.shape[0] < X_Acceleration.shape[0] :
        last_row = Force_Applied.iloc[[-1]].copy()
        while Force_Applied.shape[0] < X_Acceleration.shape[0]:
            Force_Applied = pd.concat([Force_Applied, last_row], ignore_index=True)
    
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
    
    # Add the Force_Applied column
    Dataset['Force_Applied'] = Force_Applied
    
    # Construct the new file name
    new_file_name = f"Combined_Simu_{Split_File_Name[1]}.csv"
    
    # Save the combined DataFrame to a new CSV file
    Dataset.to_csv(os.path.join(Raw_Data_Path, new_file_name), index=False)

# %% Now Concatnate each individual CSV to create the whole Dataset
Path_Combine_Individual_Simulation_Result = ('/Users/mukul/Desktop/DLR_Internship/Actual_Data/One')

# Get the List of aceeleration files in a list using glob
Individual_Simulation_Result = glob.glob(os.path.join(Path_Combine_Individual_Simulation_Result, 'Combined_Simu_*.csv'))

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

Concatenated_Simulation_Result.to_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/One/Combine_Data.csv', index=False)
