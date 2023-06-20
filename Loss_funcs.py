# loss
import numpy as np
from scipy.special import xlogy


def mse(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2)


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred, eps = 1e-10):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    result = np.clip(np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)), eps, 1.0e6)
    return result


def binary_cross_entropy_prime(y_true, y_pred, eps=1e-10):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    result = np.clip(((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true), eps, 1.0e6)
    return result


def categorical_crossentropy(y_true, y_pred, eps = 1e-10):
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.sum(y_true * np.log(y_pred))


def categorical_crossentropy_prime(y_true, y_pred, eps = 1e-10):
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -y_true / y_pred