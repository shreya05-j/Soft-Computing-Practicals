# Task 7: To understand and implement the basic concept of Backpropagation for training a neural network
import numpy as np

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward Pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.sigmoid(z2)
            
            # Backpropagation
            error_output = a2 - y
            d_z2 = error_output * self.sigmoid_derivative(a2)
            
            error_hidden = np.dot(d_z2, self.W2.T)
            d_z1 = error_hidden * self.sigmoid_derivative(a1)
            
            # Update weights
            self.W2 -= learning_rate * np.dot(a1.T, d_z2)
            self.b2 -= learning_rate * np.sum(d_z2, axis=0, keepdims=True)
            self.W1 -= learning_rate * np.dot(X.T, d_z1)
            self.b1 -= learning_rate * np.sum(d_z1, axis=0, keepdims=True)
            
    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return np.round(self.sigmoid(z2))

if __name__ == "__main__":
    print("--- Task 7: Backpropagation ---")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    nn = TwoLayerNN(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=5000, learning_rate=0.5)
    
    print("Predictions on XOR data after training:")
    for idx, sample in enumerate(X):
        prediction = nn.predict(sample.reshape(1, -1))
        print(f"Input: {sample} => Predicted: {prediction[0][0]}")
