from Base import Layer
import numpy as np
from tqdm import tqdm


class Conv2D(Layer):
    def __init__(self, image_shape, num_filters, filter_size, stride=(1, 1), padding_type='valid', padding_mode = 'constant',
                 padding_values=0):

        super().__init__()
        assert stride == (1, 1), 'Other strides not yet implemented'
        # inne stridy nie wywalają błędu, ale nie działają dobrze

        # image info
        self.input_depth, self.input_height, self.input_width = image_shape

        # filter info
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding_type = padding_type
        self.padding_mode = padding_mode
        self.padding_values = padding_values

        # kernels
        # dzielone przez stałą wartość według artykułu
        #(self.filter_size ** 2)
        self.kernels = np.random.randn(self.num_filters, self.input_depth, self.filter_size, self.filter_size)

        # output
        if self.padding_type == 'valid':
          self.output = np.zeros((self.num_filters,
                                  self.input_height - self.filter_size + 1,
                                  self.input_width - self.filter_size + 1))
        else:
          self.output = np.zeros((self.num_filters, self.input_height, self.input_width))

        # biases

        self.bias = np.random.randn(*self.output.shape)

    def convolve2d(self, image, filter, stride, padding_type):

        # 28x28 no padding
        height, width = image.shape
        #filter_height, filter_width = filter.shape

        if padding_type == 'valid':

            output_height = height - filter.shape[0] + 1
            output_width = width - filter.shape[1] + 1

        elif padding_type == 'same':

            output_height = height
            output_width = width

        elif padding_type == 'full':

            output_height = height + filter.shape[0] - 1
            output_width = width + filter.shape[1] - 1

        else:
            raise ValueError('No such padding implemented')

        image = self.padding(image, padding_type)

        output = np.zeros((output_height, output_width))

        for i in range(0, output_height, stride[0]):
            for j in range(0, output_width, stride[1]):
                output[i][j] = np.sum(image[i:i+filter.shape[0], j:j+filter.shape[1]] * filter)
        # 26x26
        return output

    def padding(self, x, padding_type, mode='constant', values=0):
      dict = {'mode': mode}
      if mode == 'constant':
          dict['constant_values'] = values
      if x.ndim == 2:
          if padding_type == 'valid':
              return x
          elif padding_type == 'same':
              pad = (self.filter_size - 1) // 2
              return np.pad(x, [(pad, pad), (pad, pad)], **dict)
          elif padding_type == 'full':
              pad = self.filter_size - 1
              return np.pad(x, [(pad, pad), (pad, pad)], **dict)
      elif x.ndim == 3:
          if padding_type == 'valid':
              return x
          elif padding_type == 'same':
              pad = (self.filter_size - 1) // 2
              return np.pad(x, [(0, 0), (pad, pad), (pad, pad)], **dict)
          elif padding_type == 'full':
              pad = self.filter_size - 1
              return np.pad(x, [(0, 0), (pad, pad), (pad, pad)], **dict)
      else:
          raise ValueError("Input must have either 2 or 3 dimensions")

    def forward(self, x):
        if self.padding_type == 'valid':
            self.input = x

        elif self.padding_type == 'same':
            self.input = self.padding(x, 'same', self.padding_mode, self.padding_values)

        self.output = np.zeros_like(self.bias)

        for i in range(self.num_filters):
            for j in range(self.input_depth):
                self.output[i] += self.convolve2d(self.input[j], self.kernels[i][j], self.stride,
                                                   padding_type='valid')

        # add bias and return
        return self.output + self.bias

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels.shape)
        input_gradient = np.zeros(self.input.shape)

        # if there are different colors
        for i in range(self.num_filters):
            for j in range(self.input_depth):
                kernels_gradient[i][j] = self.convolve2d(self.input[j], np.rot90(output_gradient[i]), self.stride,
                                        'valid')
                # full musi zwrócić ten sam wymiar czyli 28x28
                input_gradient[j] += self.convolve2d(output_gradient[i], self.kernels[i][j], self.stride, 'full')

        self.kernels -= learning_rate * kernels_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def test(network, loss, x_test, y_test):
    from sklearn.metrics import accuracy_score
    test_loss = 0
    y_true, y_pred = [], []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        output = predict(network, x)
        y_pred.append(np.argmax(output))
        y_true.append(np.argmax(y))

        test_loss += loss(y, output)

    print(f'Test error: {test_loss/len(x_test)}, accuracy:{accuracy_score(y_true, y_pred)}',)


def train(network, loss, loss_prime, x_train, y_train, epochs = 100, learning_rate = 0.01, info = True):
    for e in tqdm(range(epochs)):
        train_loss = 0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            # forward
            output = predict(network, x)
            # error
            train_loss += loss(y, output)
            #print('error', error.shape)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

            if i % 10 == 0 and i != 0:
                print(f'\r Sample {i}/{len(x_train)}, error: {train_loss/i}', end="")

        train_loss /= len(x_train)

        if info:
            print(f" \n Epochs: {e + 1}/{epochs}, error={train_loss}")

