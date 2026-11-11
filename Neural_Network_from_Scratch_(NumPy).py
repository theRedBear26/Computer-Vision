import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.A1)
        
        self.W2 += self.A1.T.dot(output_delta) * self.lr
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.lr
        self.W1 += X.T.dot(hidden_delta) * self.lr
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=5000):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(np.square(y - output))
            losses.append(loss)
            self.backward(X, y, output)
        return losses

# Load Data
digits = load_digits()
X = digits.data
y = digits.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize
X_train = X_train / 16.0
X_test = X_test / 16.0

# Train
nn = NeuralNetwork(input_size=64, hidden_size=32, output_size=10, learning_rate=0.1)
losses = nn.train(X_train, y_train, epochs=3000)

# Evaluate
predictions = nn.forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes) * 100

print(f"Test Accuracy: {accuracy:.2f}%")

plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()