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


class Lrelu(Layer):
    def __init__(self, input_layer=None, in_size=None):
        self.out_size = input_layer if input_layer is not None else in_size

    def forward(self, x):
        return np.maximum(x, x * .1)

    def backward(self, out_grad):
        grad_back = np.where(out_grad > 0, out_grad, out_grad * .1)
        return grad_back

    def optimizer_step(self, learning_rate):
        pass


class Linear(Layer):
    def __init__(self, input_layer=None, in_size=None, out_size=None):
        self.in_size = input_layer if input_layer is not None else in_size
        self.out_size = out_size
        self.w = np.random.randn(in_size, out_size)
        self.vel = np.zeros((in_size, out_size))

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
        pass

    @abstractmethod
    def backward(self, x):
        pass

    @abstractmethod
    def loss(self, **args):
        pass

    @abstractmethod
    def step(self, learning_rate):
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

    def backward(self, loss_gradient):
        for layer in self.layers[::-1]:
            loss_gradient = layer.backward(loss_gradient)

    def loss(self, y, yhat):
        loss_matrix, loss_gradient = self.loss_fcn(y, yhat)
        return loss_matrix, loss_gradient

    def step(self, learning_rate):
        for layer in self.layers:
            layer.optimizer_step(learning_rate)


class Trainer():
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.steps = 0

    def optimize(self, x, y, learning_rate):
        model = self.model
        yhat = model.forward(x)
        loss_matrix, loss_gradient = model.loss(y, yhat)
        model.backward(loss_gradient)
        model.step(learning_rate)
        loss_rms = np.sqrt(np.square(loss_matrix).sum(1)).mean()
        self.losses.append(loss_rms)

    def train_n_steps(self, n, x, y):
        for _ in range(n):
            self.steps += 1
            self.optimize(x, y, learning_rate=1 / (self.steps + 100))

    def visualize(self, skip_first=0):
        plt.plot(self.losses[skip_first:])
        plt.title('Loss for model with momentum, clean software\napproaches ' +
                  str(self.losses[-1])[:5])
        plt.savefig('model_5.jpg')
        plt.show()


if __name__ == "__main__":
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    sizes = [D_in, H, D_out]

    layers = []
    layers.append(Linear(in_size=D_in, out_size=H))
    layers.append(Lrelu(in_size=H))
    layers.append(Linear(in_size=H, out_size=D_out))

    model = MultiLayerPerceptron(layers, loss_fcn=l2_loss)

    trainer = Trainer(model)
    trainer.train_n_steps(n=500, x=x, y=y)
    trainer.visualize(skip_first=300)
