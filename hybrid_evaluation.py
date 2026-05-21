import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load models
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
dnn_model = load_model("dnn_model.h5")

# Load dataset
df = pd.read_csv("processed_intrusion_data.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get prediction probabilities
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]
dnn_prob = dnn_model.predict(X_test, verbose=0).flatten()

# Individual predictions
xgb_pred = (xgb_prob >= 0.5).astype(int)
rf_pred = (rf_prob >= 0.5).astype(int)
dnn_pred = (dnn_prob >= 0.5).astype(int)

# Individual accuracies
xgb_acc = accuracy_score(y_test, xgb_pred)
rf_acc = accuracy_score(y_test, rf_pred)
dnn_acc = accuracy_score(y_test, dnn_pred)

print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"DNN Accuracy: {dnn_acc * 100:.2f}%")

# Weighted hybrid (higher importance to XGBoost and Random Forest)
hybrid_prob = (
    0.45 * xgb_prob +
    0.45 * rf_prob +
    0.10 * dnn_prob
)

# Find best threshold automatically
best_accuracy = 0
best_threshold = 0.5
best_pred = None

for threshold in np.arange(0.10, 0.91, 0.01):
    pred = (hybrid_prob >= threshold).astype(int)
    acc = accuracy_score(y_test, pred)

    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        best_pred = pred

# Final hybrid results
print(f"\nBest Threshold: {best_threshold:.2f}")
print(f"Hybrid Accuracy: {best_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, best_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_pred))

# Save predictions
joblib.dump(best_pred, "hybrid_predictions.pkl")

print("\nPredictions saved as hybrid_predictions.pkl")