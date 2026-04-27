import numpy as np

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

X = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 1]
])

y = np.array([
    [0],
    [0],
    [1],
    [1],
    [1]
])

np.random.seed(1)

weights = 2 * np.random.random((3, 1)) - 1

print("Початкові ваги синапсів:")
print(weights)

epochs = 1000000
for iter in range(epochs):
    input_layer = X
    outputs = sigmoid(np.dot(input_layer, weights))

    error = y - outputs

    adjustments = error * sigmoid(outputs, deriv=True)

    weights += np.dot(input_layer.T, adjustments)

print("\nВаги після навчання:")
print(weights)

print("\nРезультат після навчання (прогноз нейромережі):")
print(np.round(outputs, 10))