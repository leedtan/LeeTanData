#Numpy for matrix math and matplotlib for plotting loss
import numpy as np
import matplotlib.pyplot as plt


def forward(x, w1, w2):

    #BS, D_in * D_in, H = BS, H
    hidden_raw = np.matmul(x, w1)

    #BS, H = BS, H
    hidden = np.maximum(hidden_raw, 0)

    #BS, H * H, D_out = BS, D_out
    yhat = np.matmul(hidden, w2)

    #yhat for loss and prediction. hidden for backprop
    return yhat, hidden


def backward(hidden, x, loss_gradient):

    #H, BS * BS, D_out = H, D_out
    grad_w2 = np.matmul(hidden.T, loss_gradient)

    #BS, 10 * 10, H = BS, H
    grad_hidden = np.matmul(loss_gradient, w2.T)

    #BS, H = BS, H
    grad_hidden_pre_relu = grad_hidden * (hidden > 0)

    #D_in, BS * BS, H = D_in, H
    grad_w1 = np.matmul(x.T, grad_hidden_pre_relu)

    return grad_w1, grad_w2


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

#Randomly initialize network weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

#Track losses
losses = []

#Perform full-batch optimization steps
for t in range(500):

    #Decaying learning rate
    learning_rate = 1 / (t + 100)

    #Forward propagate through the network
    yhat, hidden = forward(x, w1, w2)

    #Calculate our loss matrix. Sample by y_dimension
    loss_matrix = np.square(yhat - y)
    loss_gradient = 2 * (yhat - y)

    #Backpropagate and calculate gradients
    grad_w1, grad_w2 = backward(hidden, x, loss_gradient)

    #Clip our gradients to [-1, 1]
    grad_w1, grad_w2 = [np.clip(v, -1, 1) for v in [grad_w1, grad_w2]]

    #Update the weights by a small step in the direction of the gradient
    w1 = w1 - grad_w1 * learning_rate
    w2 = w2 - grad_w2 * learning_rate

    # norm of the loss vector for each sample. Take the mean between samples
    loss_rms = np.sqrt(np.square(loss_matrix).sum(1)).mean()
    losses.append(loss_rms)

print(losses)

#Visualize our losses over time, starting after the initial training
plt.plot(losses[300:])
plt.title(
    'Loss for model with learning decay and gradient clipping\napproaches ' +
    str(losses[-1])[:5])
plt.savefig('model_2.jpg')
plt.show()
