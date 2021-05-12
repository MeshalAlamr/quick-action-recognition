import numpy as np
import pickle

# Selected actions (Note: 0 means action "A1" and so on)
actions = [0, 3, 22, 36, 33, 26, 41, 42, 58, 16]

# Load training and testing data and labels
train_data = np.load("data/NTU-RGB-D/xview/train_data_downsampled.npy")
test_data = np.load("data/NTU-RGB-D/xview/val_data_downsampled.npy")
with open("data/NTU-RGB-D/xview/train_label.pkl","rb") as f:
    train_label = pickle.load(f)
with open("data/NTU-RGB-D/xview/val_label.pkl","rb") as f:
    test_label = pickle.load(f)  
        
# Find index for selected actions
train_index = [i for i, x in enumerate(train_label[1]) if x in actions]
test_index = [i for i, x in enumerate(test_label[1]) if x in actions]

# Create & save new training and testing data
small_train_data = train_data[train_index]
np.save("data/NTU-RGB-D/xview/small_train_data.npy", small_train_data)
small_test_data = test_data[test_index]
np.save("data/NTU-RGB-D/xview/small_val_data.npy", small_test_data)

# Create & save new training and testing labels
small_train_label = np.array(train_label)
small_train_label = small_train_label[:,train_index]
small_test_label = np.array(test_label)
small_test_label = small_test_label[:,test_index]

with open('data/NTU-RGB-D/xview/small_train_label.pkl', 'wb') as f:
    pickle.dump(small_train_label, f)

with open('data/NTU-RGB-D/xview/small_val_label.pkl', 'wb') as f:
    pickle.dump(small_test_label, f)
