import matplotlib.pyplot as plt
from data import load_mnist
from network import NeuralNetwork

def main():
    *_, X_test, y_test = load_mnist()

    nn = NeuralNetwork.load("mnist_network.npz")

    outputs = nn.forward(X_test)
    predictions = outputs.argmax(axis=1)

    mistake_indices = (predictions != y_test).nonzero()[0]
    selected_indices = mistake_indices[:12]

    figure, axes = plt.subplots(3, 4, figsize=(8, 6))

    for axis, index in zip(axes.flat, selected_indices):
        axis.imshow(X_test[index].reshape(28, 28), cmap="gray")
        axis.set_title(
            f"True: {y_test[index]} | Pred: {predictions[index]}"
        )
        axis.axis("off")

    figure.suptitle(
        f"First 12 of {len(mistake_indices)} mistakes"
    )
    figure.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()