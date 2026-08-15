import numpy as np
from data import load_mnist
from network import NeuralNetwork

def evaluate(nn, images, labels):
    total_loss = 0.0
    correct_predictions = 0

    for x, label in zip(images, labels):
        target = np.zeros(10, dtype=np.float32)
        target[label] = 1.0

        output = nn.forward(x)

        total_loss += 0.5 * ((output - target) ** 2).sum()
        correct_predictions += int(output.argmax() == label)

    average_loss = total_loss / len(images)
    accuracy = correct_predictions / len(images)

    return average_loss, accuracy

def main():
    x_train, y_train, x_test, y_test = load_mnist()

    nn = NeuralNetwork()
    rng = np.random.default_rng(42)

    training_samples = 5_000
    epochs = 3
    learning_rate = 0.1

    for epoch in range(epochs):
        indices = rng.permutation(training_samples)

        total_loss = 0.0
        correct_predictions = 0

        for index in indices:
            x = x_train[index]
            label = y_train[index]

            target = np.zeros(10, dtype=np.float32)
            target[label] = 1.0

            output = nn.forward(x)

            total_loss += 0.5 * ((output - target) ** 2).sum()
            correct_predictions += int(output.argmax() == label)

            gradients = nn.backward(target)
            nn.step(gradients, learning_rate)

        average_loss = total_loss / training_samples
        accuracy = correct_predictions / training_samples

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2%}"
        )

    test_loss, test_accuracy = evaluate(nn, x_test, y_test)

    print(
        f"Test | Loss: {test_loss:.4f} | "
        f"Accuracy: {test_accuracy:.2%}"
    )

if __name__ == "__main__":
    main()