import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist
from network import NeuralNetwork

def main():
    *_, X_test, y_test = load_mnist()

    nn = NeuralNetwork.load("mnist_network.npz")

    outputs = nn.forward(X_test)
    predictions = outputs.argmax(axis=1)

    confusion = np.zeros((10, 10), dtype=np.int32)

    for true_digit, predicted_digit in zip(y_test, predictions):
        confusion[true_digit, predicted_digit] += 1

    correct_per_digit = np.diag(confusion) # cool shet
    total_per_digit = confusion.sum(axis=1)

    accuracy_per_digit = (
        correct_per_digit / total_per_digit
    )

    print("\nAccuracy by digit:")

    for digit, accuracy in enumerate(np.sort(accuracy_per_digit, descending=True)):
        print(
            f"Digit {digit}: {accuracy:.2%} "
            f"({correct_per_digit[digit]}/"
            f"{total_per_digit[digit]})"
        )

    figure, axis = plt.subplots(figsize=(8, 7))

    image = axis.imshow(confusion, cmap="Blues")

    figure.colorbar(
        image,
        ax=axis,
        label="Number of images",
    )

    axis.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Predicted digit",
        ylabel="True digit",
        title="MNIST Confusion Matrix",
    )

    threshold = confusion.max() / 2

    for true_digit in range(10):
        for predicted_digit in range(10):
            value = confusion[true_digit, predicted_digit]

            axis.text(
                predicted_digit,
                true_digit,
                value,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )

    accuracy_percent = accuracy_per_digit * 100

    accuracy_figure, accuracy_axis = plt.subplots(
        figsize=(9, 5)
    )

    accuracy_axis.bar(
        np.arange(10),
        accuracy_percent,
    )

    accuracy_axis.set(
        xticks=np.arange(10),
        xlabel="True digit",
        ylabel="Accuracy (%)",
        ylim=(0, 107), # give some room for numbers above bars
        title="Accuracy for Each MNIST Digit",
    )

    for digit, accuracy in enumerate(accuracy_percent):
        accuracy_axis.text(
            digit,
            accuracy + 1,
            f"{accuracy:.1f}%",
            ha="center",
            fontsize=9,
        )

    figure.tight_layout()
    accuracy_figure.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()