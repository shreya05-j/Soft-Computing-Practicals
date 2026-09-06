# Task 4: To create a simple single-layer neural network using Python
import numpy as np

class SingleLayerNN:
    def __init__(self, input_size, output_size=1):
        # Initialize weights and bias
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward(self, inputs):
        # Forward pass computing the weighted sum and applying the activation function
        z = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(z)

if __name__ == "__main__":
    np.random.seed(42) # For reproducibility
    
    # Create a single-layer neural network with 3 inputs
    nn = SingleLayerNN(input_size=3)
    
    # Provide an input array
    inputs = np.array([0.5, -0.2, 0.1])
    
    # Get the output
    output = nn.forward(inputs)
    
    print("--- Task 4: Single-Layer Neural Network ---")
    print(f"Inputs: {inputs}")
    print(f"Weights:\n{nn.weights}")
    print(f"Bias: {nn.bias}")
    print(f"Network Output: {output[0]:.4f}")
