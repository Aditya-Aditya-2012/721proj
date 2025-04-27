import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    # path to our custom adjacency
    path = os.path.join('./data', dataset_name, 'adj.npz')
    # load the sparse Gaussian‐kernel graph
    adj = sp.load_npz(path).tocsc()
    n_vertex = adj.shape[0]
    return adj, n_vertex
    

def load_data(dataset, len_train, len_val, pol):
    print(pol)
        # TRAIN vs VAL exactly as your CSVs
    folder = os.path.join('./data', dataset)
        # pick the first vel_*.csv you generated
    fname = f'vel_{pol}.csv'
    df = pd.read_csv(os.path.join(folder, fname))
    # split by len_train, len_val exactly as before
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    print(len(test))
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)