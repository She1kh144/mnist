import torch
from torch import nn
from torch_data import create_data_loaders
from torch_network import TorchNeuralNetwork

def create_optimizer(model, learning_rate, l2_strength):
    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim > 1:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    return torch.optim.SGD(
        [
            {
                "params": decay_parameters,
                "weight_decay": l2_strength,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
    )

def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        correct_predictions += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

@torch.inference_mode()
def evaluate(model, data_loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size
        correct_predictions += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

def main():
    torch.manual_seed(42)

    epochs = 20
    batch_size = 32
    learning_rate = 0.1
    l2_strength = 1e-4

    model_path = "mnist_torch_network.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader = create_data_loaders(batch_size=batch_size)

    model = TorchNeuralNetwork().to(device)

    loss_function = nn.CrossEntropyLoss()

    weight_parameters = [
        model.fc1.weight,
        model.fc2.weight,
        model.fc3.weight,
    ]

    bias_parameters = [
        model.fc1.bias,
        model.fc2.bias,
        model.fc3.bias,
    ]

    optimizer = create_optimizer(
        model,
        learning_rate,
        l2_strength,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=0.001,
    )

    best_validation_accuracy = float("-inf")
    best_epoch = 0

    for epoch in range(epochs):
        current_learning_rate = (optimizer.param_groups[0]["lr"])

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            valid_loader,
            loss_function,
            device,
        )

        print(
            f"Epoch {epoch + 1} | "
            f"LR: {current_learning_rate:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2%} | "
            f"Valid Loss: {validation_loss:.4f} | "
            f"Valid Accuracy: {validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                model_path,
            )

            print(
                f"  Saved new best model at epoch "
                f"{best_epoch}"
            )

        old_learning_rate = optimizer.param_groups[0]["lr"]

        scheduler.step(validation_loss)

        new_learning_rate = optimizer.param_groups[0]["lr"]

        if new_learning_rate < old_learning_rate:
            print(
                f"  Reduced learning rate to "
                f"{new_learning_rate:.4f}"
            )

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Valid Accuracy: {best_validation_accuracy:.2%}"
    )

    print(
        f"Test | Loss: {test_loss:.4f} | "
        f"Accuracy: {test_accuracy:.2%}"
    )

if __name__ == "__main__":
    main()