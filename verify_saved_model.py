from data import load_mnist
from network import NeuralNetwork
from train import evaluate

def main():
    _, _, X_test, y_test = load_mnist()

    nn = NeuralNetwork.load("mnist_relu_softmax_network.npz")
    test_loss, test_accuracy = evaluate(nn, X_test, y_test)

    print(
        f"Verified | Loss: {test_loss:.4f} | "
        f"Accuracy: {test_accuracy:.2%}"
    )

if __name__ == "__main__":
    main()