#%% 
import pandas as pd
import numpy as np
import glob
import os
import time
#%% 
Raw_Data_Path = '/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Data'

Start_Time = time.time() 
#%% Function to covert the .xlsx files to csv file for consistency

def convert_all_excel_to_csv(Raw_Data_Path):
    
    for file_name in os.listdir(Raw_Data_Path):
        
        if file_name.endswith('.xlsx'):
            
            file_path = os.path.join(Raw_Data_Path, file_name)
            
            # Load the Excel file
            excel_data = pd.read_excel(file_path, header = None)
            
            # Define the CSV file path
            csv_file_path = os.path.splitext(file_path)[0] + '.csv'
            
            # Save as CSV
            excel_data.to_csv(csv_file_path, index = False, header = False)
            
convert_all_excel_to_csv(Raw_Data_Path)

#%% Get the List of aceeleration files in a list using glob
Acceleration_Files = glob.glob(os.path.join(Raw_Data_Path, 'Simu_*_accel_*.csv' ))

#Get the List of aceeleration files in a list using glob

Strain_Files = glob.glob(os.path.join(Raw_Data_Path, 'Simu_*_export_strain.csv' ))

# Combine two list of files 
All_Files = Acceleration_Files + Strain_Files

#%%
# Reshape the dataframe
# Assuming that each block of time step values (0 to 80) has a length of 16000 (80/0.005 + 1)
# and there are 8 such blocks (128008/16000 = 8)

# Number of steps in one complete cycle (0 to 80 with step 0.005)
Steps_Per_Cycle = int(80 / 0.005) + 1 # plus 1 is added because after every 16000 values 1 row is left blank
#print(Steps_Per_Cycle)


#%%
for file in All_Files:
    
    # Extract the file name from the path
    Acceleration_File_Name = os.path.basename(file)
    
    # Remove the file extension i.e. name without .csv
    Acceleration_File_Name_Without_Extension = os.path.splitext(Acceleration_File_Name)[0]
    
    # Split the file name using underscore
    Split_File_Name = Acceleration_File_Name_Without_Extension.split('_')
    
    # Reading ead the individual acceleration file
    Each_Acceleration_File = pd.read_csv(file, header=None)
    
    # Reshape the values excluding the first column i.e. time column
    Reshaped_Values = Each_Acceleration_File[1].values.reshape(-1, Steps_Per_Cycle).T

    # Create a new dataframe with the time steps and reshaped values
    Time_Steps = Each_Acceleration_File[0].iloc[:Steps_Per_Cycle]
    
    reshaped_df = pd.DataFrame(Reshaped_Values, columns=[f'{Split_File_Name[3]}_{Split_File_Name[2]}_{i+1}' for i in range(Reshaped_Values.shape[1])])
    reshaped_df.insert(0, 'Time', Time_Steps)

    # Save the reshaped dataframe to a new CSV file
    output_file_path = f'/Users/mukul/Desktop/DLR_Internship/Actual_Data/Raw_Data/Reshaped_{Acceleration_File_Name_Without_Extension}.csv'
    reshaped_df.to_csv(output_file_path, index=False)

End_Time = time.time()

print(f"Reshaping of CSV file are completed in {End_Time-Start_Time} sec.")
