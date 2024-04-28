import numpy as np

# Load the data from file 1
data_file1 = np.load('tvecs_1.npy')

# Load the data from file 2
data_file2 = np.load('tvecs_2.npy')

# Concatenate the data with file 1 on top of file 2
concatenated_data = np.vstack((data_file1, data_file2))

# Save the concatenated data to a new .npy file
np.save('All_tvecs.npy', concatenated_data)

# Load the data from the .npy file (Try printing the data)
# all_tvecs = np.load("All_tvecs.npy")
# print("Tvecs:", all_tvecs)
