from Layers import *
from Conv2D import *

from utils import preprocess_data_conv

from Loss_funcs import *
import tensorflow



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

x_train, y_train = preprocess_data_conv(x_train, y_train, 200, labels=(0,1,2,3))
x_test, y_test = preprocess_data_conv(x_test, y_test, 100, labels=(0,1,2,3))

network = [
    Conv2D(image_shape = (1, 28, 28), num_filters = 8, filter_size = 3, stride = (1, 1), padding_type = 'same',
           padding_mode='constant'),
    ReLULayer(),
    # no padding, so shape changes
    #Reshape (filters, depth, height, width)
    # Reshape((8, 28, 28), (8 * 28 * 28, 1)),
    # Conv2D(image_shape = (8, 28, 28), num_filters = 4, filter_size = 3, stride = (1, 1), padding_type = 'valid',
    #        padding_mode='constant'),
    # ReLU(),
    FlattenLayer(),
    Dense(8 * 28 * 28, 100),
    Dropout(0.1),
    SigmoidLayer(),
    Dense(100, 4),
    SigmoidLayer()
]

train(
    network,
    categorical_crossentropy,
    categorical_crossentropy_prime,
    x_train,
    y_train,
    epochs=2,
    learning_rate=0.01
)

test(network, categorical_crossentropy, x_test, y_test)
