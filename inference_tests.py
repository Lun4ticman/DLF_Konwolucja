import numpy as np

from Layers import *
from Conv2D import Conv2D, train, predict
import pandas as pd

# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
from Loss_funcs import *
import tensorflow



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

## flatten the data from 2D to 1D (vector) as the data shape is [28x28]

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## convert to float [0.0 - 1.0]
x_train /= 255
x_test /= 255
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')


# One-hot encoding
# y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
# y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

y_train = pd.get_dummies(y_train, dtype=float)
y_test = pd.get_dummies(y_test, dtype=float)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

x_train, y_train, x_test, y_test = x_train[:500], y_train[:500], x_test[:100], y_test[:100]
# print(x_train[0].shape, y_train[0].reshape(-1, 1).shape)

y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis =-1)

print(y_train.shape, y_test.shape)

network = [
    #Convolutional((1, 28, 28), (3, 3), 5, padding = (1,1)),
    Conv2D(image_shape = (1, 28, 28), num_filters = 5, filter_size = 3, stride = (1, 1), padding_type = 'same'),
    ReLU(),
    #Reshape (filters, depth, height, width)
    # Reshape((5, 1, 28, 28), (5 * 1 * 28 * 28, 1)),
    FlattenLayer(),
    Dense(5 * 1 *28 * 28, 100),
    ReLU(),
    Dense(100, 10),
    #ReLU()
    Sigmoid()
    # SoftmaxLayer()
]


from sklearn.metrics import accuracy_score

train(
    network,
    categorical_crossentropy,
    categorical_crossentropy_prime,
    x_train,
    y_train,
    epochs=10,
    learning_rate=0.1
)