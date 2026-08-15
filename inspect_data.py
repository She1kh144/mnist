import matplotlib.pyplot as plt
from data import load_mnist

x_train, y_train, x_test, y_test = load_mnist()

print("X_train:", x_train.shape)
print("y_train:", y_train.shape)
print("X_test: ", x_test.shape)
print("y_test: ", y_test.shape)

print("Pixel range:", x_train.min(), x_train.max())
print("First label:", y_train[0])

first_image = x_train[0].reshape(28, 28)

plt.imshow(first_image, cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()