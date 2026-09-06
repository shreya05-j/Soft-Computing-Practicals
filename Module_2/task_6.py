# Task 6: To apply Gradient Descent for updating the weights of a simple neural network during training
import numpy as np

def forward(x, w):
    return x * w

def loss(y_pred, y_true):
    return (y_pred - y_true)**2

def gradient(x, y_pred, y_true):
    return 2 * (y_pred - y_true) * x

if __name__ == "__main__":
    x = 2.0
    y_true = 5.0
    w = 0.5
    learning_rate = 0.1
    epochs = 10
    
    print("--- Task 6: Gradient Descent ---")
    for epoch in range(epochs):
        y_pred = forward(x, w)
        l = loss(y_pred, y_true)
        grad = gradient(x, y_pred, y_true)
        w = w - learning_rate * grad
        print(f"Epoch {epoch+1}: Prediction = {y_pred:.4f}, Loss = {l:.4f}, Weight = {w:.4f}")
