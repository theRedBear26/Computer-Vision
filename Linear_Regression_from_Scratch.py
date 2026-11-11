import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data (y = 3x + 2 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1)

# 2. Initialize weights and bias
w = np.random.randn(1, 1)
b = np.random.randn(1, 1)
learning_rate = 0.1
epochs = 100
m = len(X)
loss_history = []

# 3. Gradient Descent Loop
for epoch in range(epochs):
    # Forward pass: Predict
    y_pred = X @ w + b
    
    # Calculate Mean Squared Error (MSE) Loss
    loss = (1/(2*m)) * np.sum((y_pred - y)**2)
    loss_history.append(loss)
    
    # Calculate gradients
    dw = (1/m) * (X.T @ (y_pred - y))
    db = (1/m) * np.sum(y_pred - y)
    
    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db

# 4. Plot the results
plt.figure(figsize=(12, 4))

# Plot 1: Loss curve
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='red')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')

# Plot 2: Fitted line
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, X @ w + b, color='red', linewidth=2, label='Fitted Line')
plt.title(f'Linear Regression Fit\n(w={w[0][0]:.2f}, b={b[0][0]:.2f})')
plt.legend()

plt.tight_layout()
plt.show()