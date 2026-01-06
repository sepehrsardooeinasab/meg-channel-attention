from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import glob
import os


def load_data(trials_address):
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            from tqdm.notebook import tqdm
            use_tqdm = True
        else:
            use_tqdm = False
    except:
        use_tqdm = False

    X, Y = [], []
    iterable = tqdm(trials_address, desc="Loading Data") if use_tqdm else trials_address
    for trial_address in iterable:
        epoch_paths = sorted(glob.glob(os.path.join(trial_address, 'data_block*')))
        for epoch_path in epoch_paths:
            epoch = pd.read_csv(epoch_path, index_col=0)
            X.append(epoch.transpose())

            # Extract subject ID and assign label (1 for ADHD 108–133, else 0)
            subject_folder = os.path.normpath(epoch_path).split(os.sep)[-3]
            subject_id = int(subject_folder[-3:])
            Y.append(1 if 108 <= subject_id <= 133 else 0)

    return np.stack(X, axis=0), np.array(Y)


def normalize(X_train, X_val, X_test, min_max=False):
    if min_max:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_val, X_test