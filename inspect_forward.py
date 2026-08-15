import numpy as np
from data import load_mnist
from network import NeuralNetwork

x_train, y_train, *_ = load_mnist()
nn = NeuralNetwork()

x = x_train[0]

z1, a1 = nn.forward_first_layer(x)

print("Input shape:     ", x.shape)
print("Weights shape:   ", nn.w1.shape)
print("Bias shape:      ", nn.b1.shape)
print("z1 shape:        ", z1.shape)
print("Activation shape:", a1.shape)
print("Activation range:", a1.min(), a1.max(), end="\n\n")

z2, a2 = nn.forward_second_layer(a1)

print("W2 shape:        ", nn.w2.shape)
print("b2 shape:        ", nn.b2.shape)
print("z2 shape:        ", z2.shape)
print("a2 shape:        ", a2.shape)
print("a2 range:        ", a2.min(), a2.max(), end="\n\n")

z3, a3 = nn.forward_output_layer(a2)

prediction = int(a3.argmax())

print("W3 shape:        ", nn.w3.shape)
print("b3 shape:        ", nn.b3.shape)
print("Output shape:    ", a3.shape)
print("Output values:   ", a3.round(4))
print("Predicted digit: ", prediction)
print("Actual digit:    ", y_train[0], end="\n\n")

target = np.zeros(10, dtype=np.float32)
target[y_train[0]] = 1.0

error = a3 - target
loss = 0.5 * np.sum(error**2)

print("Target:          ", target)
print("Error:           ", error.round(4))
print("Loss:            ", loss, end="\n\n")

dC_da3 = a3 - target
da3_dz3 = a3 * (1.0 - a3)
delta3 = dC_da3 * da3_dz3

print("dC/da3: shape    ", dC_da3.shape)
print("da3/dz3: shape   ", da3_dz3.shape)
print("dC/dz3: shape:   ", delta3.shape)
print("dC/dz3:          ", delta3.round(4), end="\n\n")

dW3 = np.outer(delta3, a2)
db3 = delta3.copy()

print("dW3 shape:       ", dW3.shape)
print("db3 shape:       ", db3.shape)
print("dW3[4, 0]:       ", dW3[4, 0])
print("Manual check:    ", delta3[4] * a2[0], end="\n\n")

dC_da2 = nn.w3.T @ delta3
da2_dz2 = a2 * (1.0 - a2)
delta2 = dC_da2 * da2_dz2

print("dC/da2 shape:   ", dC_da2.shape)
print("da2/dz2 shape:  ", da2_dz2.shape)
print("dC/dz2: shape:  ", delta2.shape)
print("dC/dz2:         ", delta2.round(4), end="\n\n")

dW2 = np.outer(delta2, a1)
db2 = delta2.copy()

print("dW2 shape:       ", dW2.shape)
print("db2 shape:       ", db2.shape)
print("dW2[0, 0]:       ", dW2[0, 0])
print("Manual check:    ", delta2[0] * a1[0], end="\n\n")

dC_da1 = nn.w2.T @ delta2
da1_dz1 = a1 * (1.0 - a1)
delta1 = dC_da1 * da1_dz1

print("dC/da1 shape:   ", dC_da1.shape)
print("da1/dz1 shape:  ", da1_dz1.shape)
print("dC/dz1: shape:  ", delta1.shape)
print("dC/dz1:         ", delta1.round(4), end="\n\n")

dW1 = np.outer(delta1, x)
db1 = delta1.copy()

print("dW1 shape:       ", dW1.shape)
print("db1 shape:       ", db1.shape)
print("dW1[0, 0]:       ", dW1[0, 0])
print("Manual check:    ", delta1[0] * x[0], end="\n\n")