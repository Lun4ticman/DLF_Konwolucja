import numpy as np

from Base import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class ReLU(Activation):
  def __init__(self):
    def relu(x):
      return np.where(x > 0, x, 0)

    def relu_prim(x):
      return np.where(x > 0, 1, 0)

    super().__init__(relu, relu_prim)


class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * (self.input > 0)
        return input_gradient


class Sigmoid(Activation):
  def __init__(self):
      def sigmoid(x):
          temp = 1 / (1 + np.exp(-x))
          return temp

      def sigmoid_prim(x):
          s = sigmoid(x)
          # temp = s * (1 - s)
          # return temp[:, 0]
          return s * (1 - s)

      super().__init__(sigmoid, sigmoid_prim)

# TODO: Test this
class TanH(Activation):
  def __init__(self):
    def tanh(x):
      return np.tanh(x)
    def tanh_prim(x):
      t = tanh(x)
      return 1 - t**2

    super().__init__(tanh, tanh_prim)


# # TODO: Test that
# class Softmax(Activation):
#   def __init__(self):
#     def softmax(x):
#       return np.exp(x) / sum(np.exp(x))
#     def softmax_prim(y):
#       softmax = self.input.reshape(-1, 1)
#       d_softmax = softmax - y
#       return d_softmax

#     super().__init__(softmax, softmax_prim)


# class Softmax(Layer):
#     def forward(self, input):
#
#         self.input = input
#
#         max_val = np.max(input, axis=1, keepdims=True) + 1e-10
#         tmp = np.exp(input - max_val)
#         self.output = tmp / (np.sum(tmp, axis=1, keepdims=True) + 1e-12)
#         return self.output
#
#     def backward(self, d_out, learning_rate):
#
#         d_input = np.zeros_like(self.input)
#
#         y = self.output
#         d_input = np.dot(d_out, y)
#         return d_input


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, input):
        self.input = input
        self.output = self.softmax(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        batch_size = self.output.shape[0]

        # Calculate Jacobian matrix
        jacobian = np.zeros((batch_size, self.output.shape[1], self.output.shape[1]))
        for i in range(batch_size):
            for j in range(self.output.shape[1]):
                for k in range(self.output.shape[1]):
                    if j == k:
                        jacobian[i, j, k] = self.output[i, j] * (1 - self.output[i, k])
                    else:
                        jacobian[i, j, k] = -self.output[i, j] * self.output[i, k]

        # Calculate input gradient using the Jacobian matrix
        input_gradient = np.matmul(output_gradient, jacobian)

        return input_gradient


class Reshape(Layer):
  def __init__(self, input_shape, output_shape):
    super().__init__()
    self.input_shape = input_shape
    self.output_shape = output_shape

  def forward(self, input):
    return np.reshape(input, self.output_shape)

  def backward(self, output_gradient, learning_rate):
    return np.reshape(output_gradient, self.input_shape)


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        # self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.rand(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # weights_gradient = np.matmul(output_gradient, self.input.T)
        # input_gradient = np.matmul(self.weights.T, output_gradient)

        biases_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * biases_gradient
        return input_gradient


# TODO: Test in network
class Dropout(Layer):
  def __init__(self, dropout):
    super().__init__()
    self.dropout = dropout

  def forward(self, input):
    self.input = input
    return self.input

  def backward(self, output, learning_rate = 1e-6):
    data = np.zeros(output.shape)
    value = 1 / self.dropout
    data.flat[np.random.choice(len(output.flatten()), int(len(data.flat) * (1 - self.dropout)), replace=False)] = value

    self.dropped = np.multiply(output, data)

    return self.dropped


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input_shape = input.shape
        output = input.reshape(-1, 1)
        self.output = output
        return output # collapse column

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.reshape(self.input_shape)
        return input_gradient

