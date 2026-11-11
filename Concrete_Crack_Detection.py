import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data (Cracks vs No Cracks)
def generate_synthetic_data(num_samples=500, img_size=64):
    X = np.zeros((num_samples, img_size, img_size, 1))
    y = np.zeros((num_samples,))
    
    for i in range(num_samples):
        img = np.random.rand(img_size, img_size) * 0.2 # Background noise
        if i >= num_samples // 2:
            # Generate a "crack" (randomized walk line)
            x, y_pos = np.random.randint(0, img_size), 0
            while y_pos < img_size - 1:
                img[y_pos, x] = 1.0
                x += np.random.choice([-1, 0, 1])
                x = np.clip(x, 0, img_size - 1)
                y_pos += 1
            y[i] = 1.0 # Class 1: Crack
        X[i, :, :, 0] = img
    return X, y

X, y = generate_synthetic_data()
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# 2. Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# Save model for the Capstone (Project 20)
model.save("crack_model.h5")
print("Model saved as crack_model.h5")

# 4. Plot Results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Crack Detection Accuracy')
plt.legend()
plt.show()