import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================
# 1. Accuracy Comparison
# ==========================================
xgb_accuracy = 87.79
rf_accuracy = 88.42
dnn_accuracy = 87.47
hybrid_accuracy = 88.63

plt.figure(figsize=(8, 5))

models = ["XGBoost", "Random Forest", "DNN", "Hybrid IDS"]
accuracies = [xgb_accuracy, rf_accuracy, dnn_accuracy, hybrid_accuracy]

bars = plt.bar(models, accuracies)

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.2,
        f"{acc:.2f}%",
        ha="center"
    )

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(80, 90)
plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches="tight")

# ==========================================
# 2. Hybrid Confusion Matrix
# ==========================================
cm = np.array([
    [1055, 0],
    [213, 640]
])

plt.figure(figsize=(6, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
).plot(cmap="Blues", colorbar=True)

plt.title("Hybrid IDS Confusion Matrix")
plt.savefig("hybrid_confusion_matrix.png", dpi=300, bbox_inches="tight")

# ==========================================
# 3. DNN Training Accuracy
# ==========================================
train_acc = [
    0.78, 0.85, 0.87, 0.88, 0.88,
    0.88, 0.884, 0.885, 0.886, 0.885,
    0.889, 0.888, 0.891, 0.889, 0.890,
    0.892, 0.893, 0.892, 0.897, 0.898
]

val_acc = [
    0.835, 0.86, 0.875, 0.89, 0.886,
    0.889, 0.889, 0.886, 0.889, 0.891,
    0.884, 0.884, 0.888, 0.882, 0.883,
    0.879, 0.888, 0.892, 0.893, 0.895
]

plt.figure(figsize=(8, 5))
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")

plt.title("DNN Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("dnn_training_accuracy.png", dpi=300, bbox_inches="tight")

# Show all figures
plt.show()

print("Files created:")
print("1. accuracy_comparison.png")
print("2. hybrid_confusion_matrix.png")
print("3. dnn_training_accuracy.png")