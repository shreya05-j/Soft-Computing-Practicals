# Task 8: To train a simple neural network model on a dataset using Keras/TensorFlow.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

if __name__ == "__main__":
    print("--- Task 8: Train simple NN model ---")
    # Create a simple dummy dataset for training
    X_train = np.random.rand(100, 5) # 100 samples, 5 features
    y_train = np.random.randint(0, 2, size=(100, 1)) # Binary target
    
    # Build a simple model
    model = Sequential([
        Dense(8, activation='relu', input_shape=(5,)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training the neural network...")
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
    print("Training complete.")
