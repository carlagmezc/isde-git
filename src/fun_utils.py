from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    n_samples = x.shape[0]
    n_train = int(n_samples * tr_fraction)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # mezcla los índices

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    xtr = x[train_idx]
    ytr = y[train_idx]
    xts = x[test_idx]
    yts = y[test_idx]

    return xtr, ytr, xts, yts
