import numpy as np
from data import load_mnist
from network import NeuralNetwork, cross_entropy

def evaluate(nn, images, labels):
    total_loss = 0.0
    correct_predictions = 0

    for x, label in zip(images, labels):
        target = np.zeros(10, dtype=np.float32)
        target[label] = 1.0

        output = nn.forward(x)

        total_loss += cross_entropy(output, target)
        correct_predictions += int(output.argmax() == label)

    average_loss = total_loss / len(images)
    accuracy = correct_predictions / len(images)

    return average_loss, accuracy

def main():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()

    nn = NeuralNetwork()
    rng = np.random.default_rng(42)

    training_samples = len(x_train)
    epochs = 10
    batch_size = 32
    learning_rate = 0.1

    model_path = "mnist_relu_softmax_network.npz"

    best_validation_accuracy = -np.inf
    best_epoch = 0

    for epoch in range(epochs):
        indices = rng.permutation(training_samples)

        total_loss = 0.0
        correct_predictions = 0

        for start in range(0, training_samples, batch_size):
            batch_indices = indices[start : start + batch_size]

            x_batch = x_train[batch_indices]
            batch_labels = y_train[batch_indices]

            targets = np.zeros(
                (len(batch_labels), 10),
                dtype=np.float32,
            )

            targets[np.arange(len(batch_labels)), batch_labels] = 1.0

            outputs = nn.forward(x_batch)

            total_loss += cross_entropy(outputs, targets)

            predictions = outputs.argmax(axis=1)
            correct_predictions += int(
                (predictions == batch_labels).sum()
            )

            gradients = nn.backward(targets)
            nn.step(gradients, learning_rate)

        average_loss = total_loss / training_samples
        accuracy = correct_predictions / training_samples

        validation_loss, validation_accuracy = evaluate(
            nn,
            x_valid,
            y_valid,
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Train Accuracy: {accuracy:.2%} | "
            f"Valid Loss: {validation_loss:.4f} | "
            f"Valid Accuracy: {validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1

            nn.save(model_path)

            print(f"  Saved new best model at epoch {best_epoch}")

    test_loss, test_accuracy = evaluate(nn, x_test, y_test)

    print(
        f"\nTest | Loss: {test_loss:.4f} | "
        f"Accuracy: {test_accuracy:.2%}"
    )

if __name__ == "__main__":
    main()