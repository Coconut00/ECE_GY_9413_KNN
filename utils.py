import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import numpy as np


def load_data(test_size, dataset, flatten=True):

    if dataset == 'mnist':
        X, Y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])

        X = X[permutation]
        Y = Y[permutation]
        X = X.reshape((X.shape[0], -1))
        train_input, test_input, train_target, test_target = train_test_split(
            X, Y, test_size=test_size
        )

        train_input = torch.from_numpy(train_input).reshape(-1,1,28,28)
        train_target = torch.from_numpy(train_target.astype(np.int8))
        test_input = torch.from_numpy(test_input).reshape(-1,1,28,28)
        test_target = torch.from_numpy(test_target.astype(np.int8))

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)
        
    return train_input, train_target, test_input, test_target