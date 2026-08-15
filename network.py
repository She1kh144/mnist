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
        z1 = self.W1 @ x + self.b1
        a1 = sigmoid(z1)

        z2 = self.W2 @ a1 + self.b2
        a2 = sigmoid(z2)

        z3 = self.W3 @ a2 + self.b3
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
        x = self.cache["x"]
        a1 = self.cache["a1"]
        a2 = self.cache["a2"]
        a3 = self.cache["a3"]

        # Output layer
        delta3 = (a3 - target) * a3 * (1.0 - a3)
        dW3 = np.outer(delta3, a2)
        db3 = delta3.copy()

        # Second hidden layer
        delta2 = (self.W3.T @ delta3) * a2 * (1.0 - a2)
        dW2 = np.outer(delta2, a1)
        db2 = delta2.copy()

        # First hidden layer
        delta1 = (self.W2.T @ delta2) * a1 * (1.0 - a1)
        dW1 = np.outer(delta1, x)
        db1 = delta1.copy()

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }