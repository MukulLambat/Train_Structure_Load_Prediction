#%% 
import pandas as pd
import numpy as np

#%%
#Force = pd.read_excel('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Force_1.xlsx', header=None)
#Force.to_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Force.csv', header=None)
#Force = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Force_1.csv', header=None)
Acc_X = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Simu_1_accel_x.csv' , header=None)
# Acc_Y = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Simu_1_accel_y.csv', header=None)
# Acc_Z = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Simu_1_accel_z.csv', header=None)
# Strain = pd.read_csv('/Users/mukul/Desktop/DLR_Internship/Actual_Data/Simu_1_export_strain.csv', header=None)

#%%
# Reshape the dataframe
# Assuming that each block of time step values (0 to 80) has a length of 16000 (80/0.005 + 1)
# and there are 8 such blocks (128008/16000 = 8)

# Number of steps in one complete cycle (0 to 80 with step 0.005)
steps_per_cycle = int(80 / 0.005) + 1
print(steps_per_cycle)
#%%
# Reshape the values excluding the time column
reshaped_values = Acc_X[1].values.reshape(-1, steps_per_cycle).T

#%%

# Create a new dataframe with the time steps and reshaped values
time_steps = Acc_X[0].iloc[:steps_per_cycle]
reshaped_df = pd.DataFrame(reshaped_values, columns=[f'Value_{i+1}' for i in range(reshaped_values.shape[1])])
reshaped_df.insert(0, 'Time', time_steps)

# Save the reshaped dataframe to a new CSV file
output_file_path = '/Users/mukul/Desktop/New/reshaped_file.csv'
reshaped_df.to_csv(output_file_path, index=False)

#import ace_tools as tools; tools.display_dataframe_to_user(name="Reshaped DataFrame", dataframe=reshaped_df)


# %%
# print(Force[15988:15992], '\n\n')
print(Acc_X[15998:16007], '\n\n')
print(Acc_X[31998:32007], '\n\n')
print(Acc_X[47998:48007], '\n\n')
print(Acc_X[63998:64007], '\n\n')
print(Acc_X[79998:80007], '\n\n')
print(Acc_X[95998:96007], '\n\n')
print(Acc_X[11198:112007], '\n\n')
print(Acc_X[12798:128007], '\n\n')


#%%
# print(Force.tail(25), '\n\n')

# #%%
# print(Acc_X.tail(25), '\n\n')

#%%
# print(Acc_X.head(3), '\n\n')
# print(Acc_X[15998:16002], '\n\n')
# print(Acc_X.tail(3), '\n\n')
# print(Force.head(3), '\n\n')
# print(Force.tail(3), '\n\n')

# #%%
# # print(Force.columns, '\n\n')
# # print(Acc_X.columns, '\n\n')
# # print(Acc_Y.columns, '\n\n')
# # print(Acc_Z.columns, '\n\n')
# # print(Strain.columns, '\n\n')
# # """There are no column name in the csv files."""

# #%%
# print(Force.shape, '\n\n')
#print(Acc_X.shape, '\n\n')
# print(Acc_Y.shape, '\n\n')
# print(Acc_Z.shape, '\n\n')
# print(Strain.shape, '\n\n')

# #%% 
# print(pd.isna(Force.value_counts().sum()))
# print(pd.isna(Acc_X.value_counts().sum()))
# print(pd.isna(Acc_Y.value_counts().sum()))
# print(pd.isna(Acc_Z.value_counts().sum()))
# print(pd.isna(Strain.value_counts().sum()))
# %%