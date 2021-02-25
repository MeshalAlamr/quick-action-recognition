import numpy
import pickle
data = numpy.load("NTU-RGB-D/xview/train_data.npy")
with open("NTU-RGB-D/xview/train_label.pkl","rb") as f:
    label = pickle.load(f)
    
