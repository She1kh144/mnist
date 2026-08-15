import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self):
        rng = np.random.default_rng(42)

        self.W1 = rng.normal(
            loc=0,
            scale=1 / np.sqrt(784), # std
            size=(16, 784)
        ).astype(np.float32)

        self.b1 = np.zeros(16, dtype=np.float32) # begin with no preference

        self.W2 = rng.normal(
            loc=0,
            scale=1 / np.sqrt(16),
            size=(16, 16)
        ).astype(np.float32)

        self.b2 = np.zeros(16, dtype=np.float32) 

        self.W3 = rng.normal(
            loc=0,
            scale=1 / np.sqrt(16),
            size=(10, 16)
        ).astype(np.float32)

        self.b3 = np.zeros(10, dtype=np.float32)

    def forward(self, x):
        z1 = x @ self.W1.T + self.b1
        a1 = sigmoid(z1)

        z2 = a1 @ self.W2.T + self.b2
        a2 = sigmoid(z2)

        z3 = a2 @ self.W3.T + self.b3
        a3 = sigmoid(z3)

        self.cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
            "a3": a3,
        }

        return a3
    
    def backward(self, target):
        x = np.atleast_2d(self.cache["x"])
        a1 = np.atleast_2d(self.cache["a1"])
        a2 = np.atleast_2d(self.cache["a2"])
        a3 = np.atleast_2d(self.cache["a3"])
        target = np.atleast_2d(target)

        batch_size = x.shape[0]

        # Output layer
        delta3 = (a3 - target) * a3 * (1.0 - a3)
        dW3 = delta3.T @ a2 / batch_size
        db3 = delta3.mean(axis=0)

        # Second hidden layer
        delta2 = (delta3 @ self.W3) * a2 * (1.0 - a2)
        dW2 = delta2.T @ a1 / batch_size
        db2 = delta2.mean(axis=0)

        # First hidden layer
        delta1 = (delta2 @ self.W2) * a1 * (1.0 - a1)
        dW1 = delta1.T @ x / batch_size
        db1 = delta1.mean(axis=0)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    def step(self, gradients, learning_rate):
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]

        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

        self.W3 -= learning_rate * gradients["W3"]
        self.b3 -= learning_rate * gradients["b3"]