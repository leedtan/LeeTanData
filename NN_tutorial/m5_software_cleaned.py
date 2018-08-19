# Numpy for matrix math and matplotlib for plotting loss
import numpy as np
import matplotlib.pyplot as plt
# Abstract Base Class
from abc import ABC, abstractmethod

MOMENTUM = 0.9


def l2_loss(y, yhat):
    loss_matrix = np.square(yhat - y)
    loss_gradient = 2 * (yhat - y)
    return loss_matrix, loss_gradient


def apply_linear_momentum(prev_momentum, grad_parameter, momentum):
    # Calculate momentum update by linear momentum method
    assert momentum <= 1 and momentum >= 0

    return prev_momentum * momentum + grad_parameter * (1 - momentum)


class Layer(ABC):
    @abstractmethod
    def __init__(self, **args):
        pass

    @abstractmethod
    def forward(self, x):
        # Forward propagate. Remember params needed for backprop
        pass

    @abstractmethod
    def backward(self, x):
        # Return gradient to input, gradient to parameters
        pass


class lrelu(Layer):
    def __init__(self, input_layer=None, in_size=None):
        self.out_size = input_layer if input_layer is not None else in_size

    def forward(self, x):
        return np.maximum(x, x * .1)

    def backward(self, out_grad):
        grad_back = np.where(out_grad > 0, out_grad, out_grad * .1)
        return grad_back, None

    def optimizer_step(self):
        pass


class linear(Layer):
    def __init__(self, input_layer=None, in_size=None, out_size=None):
        self.in_size = input_layer if input_layer is not None else in_size
        self.out_size = out_size
        self.w = np.random.randn(in_size, out_size)
        self.vel = np.zeros(in_size, out_size)

    def forward(self, x):
        self.prev_input = x
        return np.matmul(x, self.w)

    def backward(self, out_grad):
        # in_size, BS * BS, out_size = in_size, out_size
        raw_grad_w = np.matmul(self.prev_input.T, out_grad)
        self.grad_w = np.clip(raw_grad_w, -1, 1)

        # BS, out_size * out_size, in_size = BS, in_size
        return np.matmul(out_grad, self.w.T)

    def optimizer_step(self, learning_rate):
        self.vel = apply_linear_momentum(self.vel, self.grad_w, MOMENTUM)
        self.w = self.w - self.vel * learning_rate


class Model(ABC):
    @abstractmethod
    def __init__(self, layers, **args):
        pass

    @abstractmethod
    def forward(self, x):
        # Forward propagate through layers
        pass

    @abstractmethod
    def backward(self, x):
        # Backpropagate through layers.
        pass


class MultiLayerPerceptron(Model):
    def __init__(self, layers, loss_fcn):
        self.layers = layers
        self.loss_fcn = loss_fcn

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        yhat = x
        return yhat

    def loss(self, y, yhat):
        loss_matrix, loss_gradient = self.loss_fcn(y, yhat)
        return loss_matrix, loss_gradient

    def backward(self, loss_gradient):
        for layer in self.layers[:, :, -1]:
            loss_gradient = layer.backward(loss_gradient)

    def step(self, learning_rate):
        for layer in self.layers:
            layer.optimizer_step(learning_rate)


def Trainer():
    def __init(self, model):
        self.model = model
        self.losses = []

    def optimize(self, x, y, learning_rate):
        model = self.model
        yhat = model.forward(x)
        loss_matrix, loss_gradient = model.loss(y, yhat)
        model.backward(loss_gradient)
        model.step(learning_rate)
        loss_rms = np.sqrt(np.square(loss_matrix).sum(1)).mean()
        self.losses.append(loss_rms)


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

for t in range(500):

    # Decaying learning rate
    learning_rate = 1 / (t + 100)

    # Forward propagate through the network
    if NESTEROV:
        w1_prime = w1 - w1_m * learning_rate
        w2_prime = w2 - w2_m * learning_rate
        yhat, hidden = forward(x, w1_prime, w2_prime)
    else:
        yhat, hidden = forward(x, w1, w2)

    # Calculate our loss matrix. Sample by y_dimension
    loss_matrix, loss_gradient = l2_loss(y, yhat)

    # Backpropagate and calculate gradients
    grad_w1, grad_w2 = backward(hidden, x, loss_gradient)

    # Clip our gradients to [-1, 1]
    grad_w1, grad_w2 = [np.clip(v, -1, 1) for v in [grad_w1, grad_w2]]

    # Update momentum based
    w1_m = apply_linear_momentum(w1_m, grad_w1, MOMENTUM)
    w2_m = apply_linear_momentum(w2_m, grad_w2, MOMENTUM)

    # Update the weights by the momentum
    w1 = w1 - w1_m * learning_rate
    w2 = w2 - w2_m * learning_rate

    # If Nesterov Momentum is enabled, re-calculate
    # yhat and loss when checking performance.
    # Otherwise, we would be evaluating the performance of the nesterov update.
    if NESTEROV:
        yhat, _ = forward(x, w1, w2)
        loss_matrix, _ = l2_loss(y, yhat)

    # norm of the loss vector for each sample. Take the mean between samples
    loss_rms = np.sqrt(np.square(loss_matrix).sum(1)).mean()
    losses.append(loss_rms)

print(losses)

# Visualize our losses over time
plt.plot(losses[300:])
plt.title('Loss for model with momentum\napproaches ' + str(losses[-1])[:5])
plt.savefig('model_3.jpg')
plt.show()
