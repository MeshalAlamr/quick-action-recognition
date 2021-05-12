import numpy as np

# Load the data
data = np.load("train_data.npy")

# Shape = (# of videos, coordinate axes, # of frames, # of joints, # of subjects)
shape = list(data.shape)

# Halve the number of frames
shape[2] = int(shape[2]/2)

# Initialization
downsampled_data = np.zeros(shape,dtype=np.float32)

# Downsample the data, take one frame leave another
x=0
for i in range(300): 
    if i%2==0:
        downsampled_data[:,:,x,:,:] = data[:,:,i,:,:]
        x = x+1

# Save the downsampled data
np.save("train_data_downsampled.npy", downsampled_data)
