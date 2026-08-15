import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self):
        rng = np.random.default_rng(42)

        self.w1 = rng.normal(
            loc=0,
            scale=1 / np.sqrt(784), # std
            size=(16, 784)
        ).astype(np.float32)

        self.b1 = np.zeros(16, dtype=np.float32) # begin with no preference

        self.w2 = rng.normal(
            loc=0,
            scale=1 / np.sqrt(16),
            size=(16, 16)
        ).astype(np.float32)

        self.b2 = np.zeros(16, dtype=np.float32) 

        self.w3 = rng.normal(
            loc=0,
            scale=1 / np.sqrt(16),
            size=(10, 16)
        ).astype(np.float32)

        self.b3 = np.zeros(10, dtype=np.float32)

    def forward_first_layer(self, x):
        z1 = self.w1 @ x + self.b1 # '@' for matrix multiplication
        a1 = sigmoid(z1)    

        return z1, a1

    def forward_second_layer(self, a1):
        z2 = self.w2 @ a1 + self.b2
        a2 = sigmoid(z2)    

        return z2, a2

    def forward_output_layer(self, a2):
        z3 = self.w3 @ a2 + self.b3
        a3 = sigmoid(z3)    

        return z3, a3