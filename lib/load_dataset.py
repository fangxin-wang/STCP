import os
import numpy as np
import re

def parse_d_from_str(s: str) -> int:
    # Check for "top_N" format
    match = re.match(r"PEMS03_top_(\d+)", s)
    if match:
        return int(match.group(1))
    # Check for "w12" format
    elif "PEMS03_w12" in s:
        return "w12"
    else:
        return False

def load_st_dataset(dataset, seed = 1):
    #output B, N, D
    if dataset == 'PEMS04':
        data_path = os.path.join('./data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMS03':
        data_path = os.path.join('./data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMS07':
        data_path = os.path.join('./data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMS03_w12':
        # Load the w12 dataset
        data_path = './data/PEMS03/PEMS03_w12.txt'
        data = np.loadtxt(data_path)
        print(data.shape)
    elif parse_d_from_str(dataset):
        if parse_d_from_str(dataset) == "w12":
            # Load the w12 dataset
            data_path = './data/PEMS03/PEMS03_w12.txt'
        else:
            # Load the top_N dataset
            data_path = './data/PEMS03/PEMS03_top_{}.txt'.format(parse_d_from_str(dataset))
        data = np.loadtxt(data_path)
        print(data.shape)
    elif dataset == 'PEMSBAY':
        data_path = './data/PEMSBAY/pems_bay_sub.txt'
        data = np.loadtxt(data_path)
        print(data.shape)
    elif dataset == 'syn_gpvar' or dataset == 'syn_tailup':
        if dataset == 'syn_gpvar':
            path = './data/syn_data/syn_gpvar_{}.npz'.format(seed)
        else:
            path = './data/syn_data/syn_tailup_{}.npz'.format(seed)
        data_path = os.path.join(path)
        loaded_data = np.load(data_path)
        data = loaded_data['array']
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
