import numpy as np

#xview, train for A2, A10 and A26
label = np.concatenate((np.load("xview/A2/train_label.npy"), np.load("xview/A10/train_label.npy"), np.load("xview/A26/train_label.npy")), axis=0)
data = np.concatenate((np.load("xview/A2/train_data.npy"), np.load("xview/A10/train_data.npy"),np.load("xview/A26/train_data.npy")), axis=0)
data = data[:,:,:,:,0]