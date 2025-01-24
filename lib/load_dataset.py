import os
import numpy as np
import re
def check_dataset_format(dataset_name):
    pattern = r"PEMS\d+_top_\d+$"
    return bool(re.search(pattern, dataset_name))

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
    elif dataset == 'PEMS03_top_20':
        data_path = './data/PEMS03/PEMS03_top_20.txt'
        data = np.loadtxt(data_path)
        print(data.shape)
    elif dataset == 'syn_gpvar' or dataset == 'syn_tailup' or dataset == 'syn_tailup_gen':
        if dataset == 'syn_gpvar':
            path = './data/syn_data/syn_gpvar_{}.npz'.format(seed)
        elif dataset == 'syn_tailup':
            path = './data/syn_data/syn_tailup_{}.npz'.format(seed)
        elif dataset == 'syn_tailup_gen':
            path = './data/syn_data/syn_tailup_gen_{}.npz'.format(seed)
        else:
            raise ValueError
        data_path = os.path.join(path)
        loaded_data = np.load(data_path)
        data = loaded_data['array']
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
