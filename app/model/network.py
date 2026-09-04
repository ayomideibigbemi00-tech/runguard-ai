import numpy as np

# One hidden layer neural network for regression
# Layers:  Input -> Hidden (ReLU) -> Output (linear)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size=32, output_size=1, learning_rate=0.01):
        # Layer 1 weights/bias (input -> hidden), He init: scale by sqrt(2/fan_in)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        # Layer 2 weights/bias (hidden -> output)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate  # step size for gradient descent

    def forward(self, X):
        # --- FORWARD PASS ---
        self.z1 = np.dot(X, self.W1) + self.b1   # layer 1: z1 = X.W1 + b1
        self.a1 = np.maximum(0, self.z1)          # activation: ReLU(z1) = max(0, z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # layer 2 (output): z2 = a1.W2 + b2, no activation (regression)
        return self.z2

    def backward(self, X, y, y_pred):
        # --- BACKPROPAGATION (loss = MSE, gradients via chain rule) ---
        n_samples = X.shape[0]

        dZ2 = (y_pred - y) / n_samples        # dL/dz2  (MSE derivative)
        dW2 = np.dot(self.a1.T, dZ2)          # dL/dW2 = a1^T . dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # dL/db2

        dA1 = np.dot(dZ2, self.W2.T)          # error passed back into hidden layer
        dZ1 = dA1 * (self.z1 > 0)             # apply ReLU derivative (1 if z1>0 else 0)

        dW1 = np.dot(X.T, dZ1)                # dL/dW1 = X^T . dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # dL/db1

        # gradient descent update: W -= lr * dW
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=200):
        # training loop: forward pass -> backward pass, repeated
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)

    def predict(self, X):
        # inference only, no weight update
        return self.forward(X).flatten()[0]