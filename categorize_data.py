import numpy as np
import pickle

#xview, train for A2, A10 and A26
data = np.load("NTU-RGB-D/xview/train_data.npy")
with open("NTU-RGB-D/xview/train_label.pkl","rb") as f:
    label = pickle.load(f)
    for j in [1, 9, 25]:
        index = [i for i, x in enumerate(label[1]) if x == j]
        x = []
        y = []
        for i in range(len(index)):
            x.append(label[0][index[i]])
            y.append(data[index[i]])
            np.save(f"NTU-RGB-D/xview/A{j+1}/train_label.npy",x)
            np.save(f"NTU-RGB-D/xview/A{j+1}/train_data.npy",y)

# #xsub, train for A2, A10 and A26
# data = np.load("NTU-RGB-D/xsub/train_data.npy")
# with open("NTU-RGB-D/xsub/train_label.pkl","rb") as f:
#     for j in [1, 9, 25]:
#         label = pickle.load(f)
#         index = [i for i, x in enumerate(label[1]) if x == j]
#         x = []
#         y = []
#         for i in range(len(index)):
#             x.append(label[0][index[i]])
#             y.append(data[index[i]])
#             np.save(f"NTU-RGB-D/xsub/A{j+1}/train_label.npy",x)
#             np.save(f"NTU-RGB-D/xsub/A{j+1}/train_data.npy",y)
            
# #xview, val for A2, A10 and A26
# data = np.load("NTU-RGB-D/xview/train_data.npy")
# with open("NTU-RGB-D/xview/train_label.pkl","rb") as f:
#     for j in [1, 9, 25]:
#         label = pickle.load(f)
#         index = [i for i, x in enumerate(label[1]) if x == j]
#         x = []
#         y = []
#         for i in range(len(index)):
#             x.append(label[0][index[i]])
#             y.append(data[index[i]])
#             np.save(f"NTU-RGB-D/xview/A{j+1}/train_label.npy",x)
#             np.save(f"NTU-RGB-D/xview/A{j+1}/train_data.npy",y)

# #xview, train for A2, A10 and A26
# data = np.load("NTU-RGB-D/xview/train_data.npy")
# with open("NTU-RGB-D/xview/train_label.pkl","rb") as f:
#     for j in [1, 9, 25]:
#         label = pickle.load(f)
#         index = [i for i, x in enumerate(label[1]) if x == j]
#         x = []
#         y = []
#         for i in range(len(index)):
#             x.append(label[0][index[i]])
#             y.append(data[index[i]])
#             np.save(f"NTU-RGB-D/xview/A{j+1}/train_label.npy",x)
#             np.save(f"NTU-RGB-D/xview/A{j+1}/train_data.npy",y)
