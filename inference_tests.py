from Layers import *
from Conv2D import *

from utils import preprocess_data_conv

from Loss_funcs import *
import tensorflow



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

x_train, y_train = preprocess_data_conv(x_train, y_train, 200)
x_test, y_test = preprocess_data_conv(x_test, y_test, 100)

network = [
    #Convolutional((1, 28, 28), (3, 3), 5, padding = (1,1)),
    Conv2D(image_shape = (1, 28, 28), num_filters = 5, filter_size = 3, stride = (1, 1), padding_type = 'same',
           padding_mode='maximum'),
    ReLULayer(),
    # no padding, so shape changes
    #Reshape (filters, depth, height, width)
    Reshape((5, 28, 28), (5 * 28 * 28, 1)),
    # FlattenLayer(),
    Dense(5 * 28 * 28, 100),
    SigmoidLayer(),
    Dense(100, 2),
    #ReLU()
    SigmoidLayer()
]

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=5,
    learning_rate=0.01
)

test(network, binary_cross_entropy, x_test, y_test)
