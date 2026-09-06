# Task 2: To implement the weighted sum of inputs and bias used in an artificial neuron
import numpy as np

def compute_weighted_sum(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

if __name__ == "__main__":
    inputs = np.array([1.5, 2.0, -1.0])
    weights = np.array([0.5, -0.2, 0.8])
    bias = 0.1
    
    z = compute_weighted_sum(inputs, weights, bias)
    
    print("--- Task 2: Weighted Sum ---")
    print(f"Inputs: {inputs}")
    print(f"Weights: {weights}")
    print(f"Bias: {bias}")
    print(f"Weighted Sum: {z:.2f}")
