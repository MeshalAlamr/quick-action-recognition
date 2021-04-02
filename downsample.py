import numpy as np

data = np.load("val_data.npy")
#data = data[:,:,:,:,0]

# shape = (n_videos, xyz, frames, joints, subjects)
shape = list(data.shape)
shape[2] = int(shape[2]/2) #halve the number of frames

# initialize the new data
downsampled_data = np.zeros(shape,dtype=np.float32)

# fill the new data, take one frame leave another
x=0
for i in range(300): 
    if i%2==0:
        downsampled_data[:,:,x,:,:] = data[:,:,i,:,:]
        x = x+1

np.save("val_data_downsampled.npy",downsampled_data)
