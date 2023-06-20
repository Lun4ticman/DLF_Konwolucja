import numpy as np

import tensorflow


def preprocess_data_conv(x, y, limit, shape=(1, 28, 28)):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    # x = x.reshape(len(x), 1, 28, 28)
    x = x.reshape(len(x), *shape)
    x = x.astype("float32") / 255
    y = tensorflow.keras.utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y


def preprocess_data_dense(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = tensorflow.keras.utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]
