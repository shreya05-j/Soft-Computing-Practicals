# Task 5: To create a Multilayer Perceptron (MLP) with input, hidden, and output layers using Keras/TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_mlp():
    # Sequential model groups a linear stack of layers
    model = Sequential([
        # Input layer is implicitly defined by the input_dim argument of the first hidden layer.
        Dense(16, input_dim=8, activation='relu', name='Hidden_Layer_1'),
        Dense(8, activation='relu', name='Hidden_Layer_2'),
        Dense(1, activation='sigmoid', name='Output_Layer')
    ])
    return model

if __name__ == "__main__":
    print("--- Task 5: Multilayer Perceptron (MLP) ---")
    model = create_mlp()
    model.summary()
