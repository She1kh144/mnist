import torch
from torch.utils.data import DataLoader, TensorDataset
from data import load_mnist

def create_data_loaders(batch_size=32):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()

    train_dataset = TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
    valid_dataset = TensorDataset(torch.as_tensor(x_valid, dtype=torch.float32), torch.as_tensor(y_valid, dtype=torch.long))
    test_dataset = TensorDataset(torch.as_tensor(x_test, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.long))

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader