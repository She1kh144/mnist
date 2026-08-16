import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist(data_dir: str = "data"):
    """Load the MNIST dataset from OpenML and return training and test sets."""
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=data_dir
    )

    images = mnist.data.astype(np.float32)
    labels = mnist.target.astype(np.int64)

    images = images / 255.0  # Normalize pixel values to [0, 1]

    x_train, y_train = images[:50000], labels[:50000]
    x_valid, y_valid = images[50000:60000], labels[50000:60000]
    x_test, y_test = images[60000:], labels[60000:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test