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
print("Activation range:", a1.min(), a1.max())

z2, a2 = nn.forward_second_layer(a1)

print("\nW2 shape:      ", nn.w2.shape)
print("b2 shape:        ", nn.b2.shape)
print("z2 shape:        ", z2.shape)
print("a2 shape:        ", a2.shape)
print("a2 range:        ", a2.min(), a2.max())

z3, a3 = nn.forward_output_layer(a2)

prediction = int(a3.argmax())

print("\nW3 shape:      ", nn.w3.shape)
print("b3 shape:        ", nn.b3.shape)
print("Output shape:    ", a3.shape)
print("Output values:   ", a3.round(4))
print("Predicted digit: ", prediction)
print("Actual digit:    ", y_train[0])