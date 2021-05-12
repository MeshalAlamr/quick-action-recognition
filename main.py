import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pandas as pd

#########################################################
torch.cuda.empty_cache()

batch_size = 4 
num_epoch = 80 
num_class = 60 
base_lr = 0.1
lr = base_lr
dropout = 0.5
step = [10, 50]
weight_decay = 0.0001
experiment = "stgcn_60_actions_small_9"
resultsFolder = "results/"
path = "data/NTU-RGB-D/xview/"
features_train = torch.FloatTensor(np.array(np.load(path + "train_data_downsampled.npy")))
features_test = torch.FloatTensor(np.array(np.load(path + "val_data_downsampled.npy")))
labels_train = np.array(pickle.load(open(path + 'train_label.pkl', 'rb'))[1][:])
labels_test = np.array(pickle.load(open(path + 'val_label.pkl', 'rb'))[1][:])
resultsFile = "stgcn_60_actions_small_9.csv"
state_path = resultsFolder + experiment + ".pth"

#########################################################

# In case of using small data, uncomment this section

#num_class = len(np.unique(labels_train))

# Convert to int
# labels_train = np.array(list(map(int, labels_train)))
# labels_test = np.array(list(map(int, labels_test)))

# Relabel 
# actions = np.unique(labels_train)
# for j, k in enumerate(actions):
#     index = [i for i, x in enumerate(labels_test) if x == k]
#     labels_test[index] = j
#     index = [i for i, x in enumerate(labels_train) if x == k]
#     labels_train[index] = j

#################### Sampler

class_sample_count = np.array([len(np.where(labels_train == t)[0]) for t in np.unique(labels_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[int(t)] for t in labels_train])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

####################### Trainloader and testloader

labels_train = torch.LongTensor(labels_train)
train_dataset = torch.utils.data.TensorDataset(features_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=0, sampler=sampler,pin_memory =True)
del features_train, labels_train
labels_test = torch.LongTensor(labels_test)
test_dataset = torch.utils.data.TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, num_workers=0, pin_memory =True)
del features_test, labels_test

def get_edge():
    num_node = 25
    self_link = [(i, i) for i in range(num_node)]
    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                      (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                      (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                      (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                      (22, 23), (23, 8), (24, 25), (25, 12)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    edge = self_link + neighbor_link
    center = 21 - 1
    return (edge, center)


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def get_adjacency(hop_dis, center, num_node, max_hop, dilation):
    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_node, num_node))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = adjacency
    A = []
    for hop in valid_hop:
        a_root = np.zeros((num_node, num_node))
        a_close = np.zeros((num_node, num_node))
        a_further = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, center] == hop_dis[
                        i, center]:
                        a_root[j, i] = normalize_adjacency[j, i]
                    elif hop_dis[j,
                                 center] > hop_dis[i,
                                                   center]:
                        a_close[j, i] = normalize_adjacency[j, i]
                    else:
                        a_further[j, i] = normalize_adjacency[j, i]
        if hop == 0:
            A.append(a_root)
        else:
            A.append(a_root + a_close)
            A.append(a_further)
    A = np.stack(A)
    return (A)


layout = 'ntu-rgb+d',
strategy = 'spatial'
max_hop = 1
dilation = 1
num_node = 25
edge, center = get_edge()
hop_dis = get_hop_distance(num_node, edge, max_hop=max_hop)
A = get_adjacency(hop_dis, center, num_node, max_hop, dilation)
A = torch.tensor(A, dtype=torch.float32, requires_grad=False)


#######################################################################

class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


######################################################################

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


######################################################################

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, A,
                 edge_importance_weighting, dropout):
        super().__init__()
        
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9 
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, dropout=dropout, residual=False),
            st_gcn(64, 64, kernel_size, 1, dropout=dropout),
            st_gcn(64, 64, kernel_size, 1, dropout=dropout),
            st_gcn(64, 128, kernel_size, 2, dropout=dropout),
            st_gcn(128, 128, kernel_size, 1, dropout=dropout),
            st_gcn(128, 256, kernel_size, 2, dropout=dropout),
            st_gcn(256, 256, kernel_size, 1, dropout=dropout),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


#########################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#########################################################################

def accuracy(output, labels):
    output = F.log_softmax(output, dim=1)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


##########################################################################


def train(epoch, history_train, base_lr, step):
    model.train()

    lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #########################################
    acc_avg = 0
    loss_avg = 0
    count = 0

    for i, c in enumerate(train_loader):
        features_batch = c[0].cuda()
        labels_batch = c[1].cuda()
        optimizer.zero_grad()
        output = model(features_batch)
        loss_train = loss(output, labels_batch)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        loss_avg += float(loss_train.cuda().item())
        acc_train = float(accuracy(output, labels_batch).cuda().item())
        acc_avg += acc_train
        count += 1

        loss_avg = loss_avg / count
    acc_avg = acc_avg / count
    history_train = history_train.append({'epoch': epoch, 'loss': loss_avg, 'acc': acc_avg}, ignore_index=True)
    history_train.to_csv(resultsFolder + 'train_' + resultsFile, index=False)
    return history_train


def test(epoch, history_test):
    acc_avg = 0
    loss_avg = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for i, c in enumerate(test_loader):
            features_batch = c[0].cuda()
            labels_batch = c[1].cuda()
            output = model(features_batch)
            loss_avg += float(loss(output, labels_batch).cuda().item())
            # loss_avg += F.nll_loss(output, labels_batch)
            acc_avg += float(accuracy(output, labels_batch).cuda().item())
            count += 1
    loss_avg = loss_avg / count
    acc_avg = acc_avg / count
    history_test = history_test.append({'epoch': epoch, 'loss': loss_avg, 'acc': acc_avg}, ignore_index=True)
    history_test.to_csv(resultsFolder + 'test_' + resultsFile, index=False)
    return history_test


######################################################################

model = Model(in_channels=3, num_class=num_class, A=A,
              edge_importance_weighting=True, dropout=dropout)

seed = 42
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(seed)
    A = A.cuda()
    model = model.cuda()

model.apply(weights_init)
loss = nn.CrossEntropyLoss()

# Adam
optimizer = optim.Adam(
    model.parameters(),
    lr=base_lr)
optimizer = optim.SGD(
    model.parameters(),
    lr=base_lr,
    momentum=0.9,
    nesterov=True,
    weight_decay=weight_decay)

########################################################

history_train = pd.DataFrame({'epoch': [], 'loss': [], 'acc': []})
history_test = pd.DataFrame({'epoch': [], 'loss': [], 'acc': []})
for i in tqdm(range(num_epoch)):
    history_train = train(i, history_train, base_lr, step)
    history_test = test(i, history_test)
    torch.save(model.state_dict(), state_path)
