# Task 3: To implement and visualize Sigmoid, Tanh, and ReLU activation functions using Python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

if __name__ == "__main__":
    # Generate input values
    x = np.linspace(-10, 10, 100)

    # Plot the activation functions
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(x, sigmoid(x), color='blue')
    plt.title("Sigmoid Activation Function")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(x, tanh(x), color='red')
    plt.title("Tanh Activation Function")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(x, relu(x), color='green')
    plt.title("ReLU Activation Function")
    plt.grid(True)

    plt.tight_layout()
    
    # Save the plots
    plt.savefig("activation_functions_task3.png")
    print("--- Task 3: Activation Functions ---")
    print("Plots saved as activation_functions_task3.png")
    # plt.show()
