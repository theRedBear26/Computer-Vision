import numpy as np
import matplotlib.pyplot as plt

# 1. XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. Network Architecture: 2 Inputs -> 4 Hidden -> 1 Output
np.random.seed(42)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

learning_rate = 0.5
epochs = 10000
losses = []

# 3. Training Loop
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Loss (Mean Squared Error)
    loss = np.mean((A2 - y)**2)
    losses.append(loss)
    
    # Backpropagation
    dA2 = A2 - y
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# 4. Results
print("Predictions after training:")
print(A2.round(2))
print(f"Final Loss: {losses[-1]:.4f}")

plt.plot(losses)
plt.title("XOR Neural Network Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()