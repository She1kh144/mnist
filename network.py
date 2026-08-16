import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """Compute the ReLU function."""
    return np.maximum(0.0, x)

def softmax(x):
    """Compute the softmax function."""
    # Shift the input to avoid numerical instability
    shifted_x = x - np.max(
        x,
        axis=-1,
        keepdims=True,
    )

    exponentials = np.exp(shifted_x)

    return exponentials / exponentials.sum(
        axis=-1,
        keepdims=True,
    )

def cross_entropy(output, target):
    """Compute the cross-entropy loss."""
    safe_output = np.clip(output, 1e-12, 1.0)
    return -np.sum(target * np.log(safe_output))

class NeuralNetwork:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)

        input_size = 784
        hidden_size_1 = 64
        hidden_size_2 = 32
        output_size = 10

        self.W1 = (
            rng.standard_normal((hidden_size_1, input_size))
            * np.sqrt(2.0 / input_size)
        ).astype(np.float32)

        self.b1 = np.zeros(hidden_size_1, dtype=np.float32)

        self.W2 = (
            rng.standard_normal((hidden_size_2, hidden_size_1))
            * np.sqrt(2.0 / hidden_size_1)
        ).astype(np.float32)

        self.b2 = np.zeros(hidden_size_2, dtype=np.float32)

        self.W3 = (
            rng.standard_normal((output_size, hidden_size_2))
            * np.sqrt(1.0 / hidden_size_2)
        ).astype(np.float32)

        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        z1 = x @ self.W1.T + self.b1
        a1 = relu(z1)

        z2 = a1 @ self.W2.T + self.b2
        a2 = relu(z2)

        z3 = a2 @ self.W3.T + self.b3
        a3 = softmax(z3)

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
    
    def backward(self, target, l2_strength=0.0):
        x = np.atleast_2d(self.cache["x"])
        z1 = np.atleast_2d(self.cache["z1"])
        z2 = np.atleast_2d(self.cache["z2"])
        a1 = np.atleast_2d(self.cache["a1"])
        a2 = np.atleast_2d(self.cache["a2"])
        a3 = np.atleast_2d(self.cache["a3"])
        target = np.atleast_2d(target)

        batch_size = x.shape[0]

        # Output layer
        delta3 = a3 - target # softmax derivative combined with cross-entropy loss
        dW3 = delta3.T @ a2 / batch_size
        db3 = delta3.mean(axis=0)

        # Second hidden layer
        delta2 = (delta3 @ self.W3) * (z2 > 0)
        dW2 = delta2.T @ a1 / batch_size
        db2 = delta2.mean(axis=0)

        # First hidden layer
        delta1 = (delta2 @ self.W2) * (z1 > 0)
        dW1 = delta1.T @ x / batch_size
        db1 = delta1.mean(axis=0)

        dW1 += l2_strength * self.W1
        dW2 += l2_strength * self.W2
        dW3 += l2_strength * self.W3

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

    def save(self, file_path):
        np.savez(
            file_path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
        )

    @classmethod
    def load(cls, file_path):
        nn = cls()

        with np.load(file_path) as parameters:
            nn.W1 = parameters["W1"]
            nn.b1 = parameters["b1"]
            nn.W2 = parameters["W2"]
            nn.b2 = parameters["b2"]
            nn.W3 = parameters["W3"]
            nn.b3 = parameters["b3"]

        return nn