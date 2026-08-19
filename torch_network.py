import math
import torch
from torch import nn

class TorchNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

        self.relu = nn.ReLU()

        self.initialize_parameters()

    def initialize_parameters(self): # He init for fc1 and fc2 for relu
        nn.init.kaiming_normal_(
            self.fc1.weight,
            nonlinearity="relu",
        )
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(
            self.fc2.weight,
            nonlinearity="relu",
        )
        nn.init.zeros_(self.fc2.bias)

        nn.init.normal_(
            self.fc3.weight,
            mean=0.0,
            std=math.sqrt(1.0 / self.fc3.in_features),
        )
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.reshape(x.shape[0], 784)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        logits = self.fc3(x)

        return logits