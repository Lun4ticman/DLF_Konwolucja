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


class TanH(Activation):
  def __init__(self):
    def tanh(x):
      return np.tanh(x)
    def tanh_prim(x):
      t = tanh(x)
      return 1 - t**2

    super().__init__(tanh, tanh_prim)


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.scores = None

    def forward(self, inputs):
        self.inputs = inputs
        exp_scores = np.exp(inputs - np.max(inputs))
        self.scores = exp_scores / exp_scores.sum()
        return self.scores

    def backward(self, grad, learning_rate):
        d_softmax = (
                np.diag(self.scores)
                - self.scores.T @ self.scores)
        # grad_inputs = np.dot(grad.T, (np.diag(self.scores) - np.outer(self.scores, self.scores)))
        input_grad = grad @ d_softmax
        return input_grad #input_grad.reshape(self.inputs.shape)


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
        # according to an article weights should be different

        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2/input_size)
        self.bias = np.random.rand(output_size, 1) * np.sqrt(2/1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient


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

    dropped = np.multiply(output, data)

    return dropped


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input_shape = input.shape
        output = input.reshape(-1, 1)
        self.output = output
        return output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.reshape(self.input_shape)
        return input_gradient


class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input):
        self.input = input
        self.output = self.sigmoid(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        sigmoid_derivative = self.output * (1 - self.output)
        input_gradient = sigmoid_derivative * output_gradient
        return input_gradient

