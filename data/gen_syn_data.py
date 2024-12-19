import sys
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.linalg import cholesky
from statsmodels.tsa.stattools import adfuller
import configparser
from lib.TrainInits import init_seed

import argparse

#parser
parser = argparse.ArgumentParser( )

parser.add_argument('--syn_seed', default=1007, type=int)
parser.add_argument('--ar', action="store_true", default=False)
parser.add_argument('--ma', action="store_true", default=False)
parser.add_argument("--hw", action="store_true", default=False)
parser.add_argument("--corr", action="store_true", default=False)
parser.add_argument("--node_num", default = 5, type = int)
parser.add_argument("--T_num", default = 20000, type = int)

args = parser.parse_args()

init_seed(args.syn_seed)

# Function to generate ARIMA series
def generate_arima_series(T, ar_params, ma_params, sigma=1):
    ar = np.r_[1, -np.array(ar_params)]  # AR params (starts with 1 for lag 0)
    ma = np.r_[1, np.array(ma_params)]  # MA params (starts with 1 for lag 0)
    arma_process = ArmaProcess(ar, ma)
    return arma_process.generate_sample(T, scale=sigma)

def test_stationary(time_series):
    bad_cnt = 0
    for ts in time_series:
        adf_test = adfuller(ts)
        if adf_test[1] >= 0.05:
            bad_cnt+=1

    if bad_cnt:
        print(bad_cnt,'Non-stationary TS out of',time_series.shape[0])
    else:
        print('All stationary!')
    return

# Function to generate Holt-Winters series
def generate_holtwinters_series(T, seasonal_periods, trend='add', seasonal='add', alpha=0.2, beta=0.1, gamma=0.1):
    # Generate random data for initialization
    data = np.random.randn(T) + np.linspace(0, 10, T)
    model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
    return fitted_model.fittedvalues


# Function to generate multivariate time series with correlation
def generate_multivariate_time_series(T, N, data_generating_process, correlation_matrix=None, **kwargs):
    time_series_data = []

    for i in range(N):
        if data_generating_process[i] == 'ARIMA':
            ts = generate_arima_series(T, kwargs['ar_params'][i], kwargs['ma_params'][i])
        elif data_generating_process[i] == 'Holt-Winters':
            ts = generate_holtwinters_series(T, kwargs['seasonal_periods'][i])
        else:
            ts = np.random.randn(T)  # Default to random normal series if unspecified
        time_series_data.append(ts)

    time_series_data = np.array(time_series_data)

    # Apply correlation if correlation matrix is provided
    if correlation_matrix is not None:
        # Perform Cholesky decomposition
        L = cholesky(correlation_matrix, lower=True)
        time_series_data = np.dot(L, time_series_data)

    return time_series_data.T  # Return as T * N shape

def generate_random_correlation_matrix(n):
    # Step 1: Generate a random matrix
    #A = np.random.randn(n, n)

    A = np.random.choice( [0,0.5,1], size = (n,n), p=[0.5,0.4,0.1])
    print(A)

    # Step 2: Create a symmetric positive semi-definite matrix (covariance matrix)
    cov_matrix = np.dot(A, A.T)

    # Step 3: Normalize to create a correlation matrix
    D = np.diag(1 / (np.sqrt(np.diag(cov_matrix)) + 0.00001)   )  # Diagonal matrix with 1/sqrt(diagonal)
    print(D)
    correlation_matrix = np.dot(np.dot(D, cov_matrix), D)  # D * cov_matrix * D

    print(correlation_matrix)
    return correlation_matrix



def generate_config_file(N, correlation_matrix, config_path):
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    if correlation_matrix is None:
        raise ValueError("correlation_matrix is None")

    config['data'] = {
        'num_nodes': N,
        'lag': 12,
        'horizon': 1,
        'val_ratio' : '0.2',
        'test_ratio' : '0.2',
    'tod' : False, 'normalizer' : 'std', 'column_wise' : False, 'default_graph' : True}

    config['model'] = {
        'input_dim': '1',
        'output_dim': '1',
        'embed_dim': '10',
        'rnn_units': '64',
        'num_layers': '2',
        'cheb_order': '2',
        'p1': '0.1'
    }

    # [train] section
    config['train'] = {
        'loss_func': 'mae',
        'seed': '10',
        'batch_size': '64',
        'epochs': '100',
        'lr_init': '0.003',
        'lr_decay': 'False',
        'lr_decay_rate': '0.3',
        'lr_decay_step': '5,20,40,70',
        'early_stop': 'True',
        'early_stop_patience': '15',
        'grad_norm': 'False',
        'max_grad_norm': '5',
        'real_value': 'True'
    }

    # [test] section
    config['test'] = {
        'mae_thresh': 'None',
        'mape_thresh': '0.001'
    }

    # [log] section
    config['log'] = {
        'log_step': '20',
        'plot': 'False'
    }

    config['cor_m'] = {
        'matrix': np.array2string(correlation_matrix)
    }

    if os.path.exists(config_path):
        print('Already Exists Config')
    else:
        with open( config_path , 'w') as configfile:
            config.write(configfile)
        print('Write Config')


T = args.T_num  # Number of time points
N = args.node_num  # Number of time series

if args.hw:
    gen_type = ['ARIMA', 'Holt-Winters']
else:
    gen_type = ['ARIMA']

data_generating_process = np.random.choice(gen_type, N)

ar_params, ma_params, seasonal_periods = [], [], []
for ts in data_generating_process:
    if ts == 'ARIMA':
        corr = np.random.rand(2)
        if args.ar:
            ar_params.append( [corr[0]] )
        else:
            ar_params.append( [0] )
        if args.ma:
            ma_params.append( [corr[1]] )
        else:
            ma_params.append( [0] )
        seasonal_periods.append(None)
    elif ts == 'Holt-Winters':
        ss_p = np.random.randint(20) + 1
        seasonal_periods.append(ss_p)
        ar_params.append([])
        ma_params.append([])

# ar_params = [[0.9], [], [0.5]]  # ARIMA specific parameters for each series
# ma_params = [[0.2], [], [0.3]]
# seasonal_periods = [None, 12, None]
seed = args.syn_seed
if args.corr:
    correlation_matrix = generate_random_correlation_matrix(N)
else:
    correlation_matrix = None

# Generate time series
time_series = generate_multivariate_time_series(T, N, data_generating_process,
                                                correlation_matrix=correlation_matrix,
                                                ar_params=ar_params,
                                                ma_params=ma_params,
                                                seasonal_periods=seasonal_periods)

test_stationary(time_series)
print(time_series.shape)

path = './data/syn_data/syn_arima_{}.npz'.format(seed)
config_path = './model/syn_arima_{}.conf'.format(seed)

generate_config_file(N, correlation_matrix, config_path)

if os.path.exists(path):
    print('Already Exists Dataset')
else:
    np.savez(path, array=time_series)
    print('Created New Dataset')
