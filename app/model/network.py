import numpy as np

class NeuralNetwork:
    """A simple feedforward neural network built from scratch using NumPy."""
    def __init__(self, input_size, hidden_size=32, output_size=1, learning_rate=0.01):
        # He initialization to prevent vanishing/exploding gradients
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1  # (n_samples, hidden_size)
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (n_samples, output_size)
        return self.z2

    def backward(self, X, y, y_pred):
        n_samples = X.shape[0]
        dZ2 = (y_pred - y) / n_samples
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.z1 > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=200):
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)

    def predict(self, X):
        return self.forward(X).flatten()[0]