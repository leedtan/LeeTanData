# Numpy for matrix math and matplotlib for plotting loss
import numpy as np
import matplotlib.pyplot as plt


def l2_loss(y, yhat):
    loss_matrix = np.square(yhat - y)
    loss_gradient = 2 * (yhat - y)
    return loss_matrix, loss_gradient


def apply_linear_momentum(prev_momentum, grad_parameter, momentum):
    # Calculate momentum update by linear momentum method
    assert momentum <= 1 and momentum >= 0

    return prev_momentum * momentum + grad_parameter * (1 - momentum)


def forward(x, w1, w2):

    # BS, D_in * D_in, H = BS, H
    hidden_raw = np.matmul(x, w1)

    # BS, H = BS, H
    hidden = np.maximum(hidden_raw, hidden_raw * .1)

    # BS, H * H, D_out = BS, D_out
    yhat = np.matmul(hidden, w2)

    # yhat for loss and prediction. hidden for backprop
    return yhat, hidden


def backward(hidden, x, loss_gradient):

    # H, BS * BS, D_out = H, D_out
    grad_w2 = np.matmul(hidden.T, loss_gradient)

    # BS, 10 * 10, H = BS, H
    grad_hidden = np.matmul(loss_gradient, w2.T)

    # BS, H = BS, H
    grad_hidden_pre_relu = np.where(grad_hidden > 0, grad_hidden,
                                    grad_hidden * .1)

    # D_in, BS * BS, H = D_in, H
    grad_w1 = np.matmul(x.T, grad_hidden_pre_relu)

    return grad_w1, grad_w2


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize network weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# Zero initialize network momentum
w1_m = np.zeros((D_in, H))
w2_m = np.zeros((H, D_out))

# Track losses
losses = []

# Perform full-batch optimization steps

MOMENTUM = 0.9
NESTEROV = True

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
plt.title('Loss for model with nesterov momentum\napproaches ' +
          str(losses[-1])[:5])
plt.savefig('model_4.jpg')
plt.show()
