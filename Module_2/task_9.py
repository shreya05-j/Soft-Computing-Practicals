# Task 9: To evaluate the performance of a trained neural network using suitable evaluation metrics.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

if __name__ == "__main__":
    print("--- Task 9: Evaluate model performance ---")
    # Simulated true labels and predicted probabilities from a trained model
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_pred_prob = np.array([0.1, 0.9, 0.8, 0.4, 0.85, 0.2, 0.6, 0.95, 0.7, 0.3])
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("--- Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
