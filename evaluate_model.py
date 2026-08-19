import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# 1. Load Dataset

df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))

# 2. Load Saved Model & Vectorizer

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")

# 3. Same Train/Test Split

X = df["Word"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Convert Test Data to TF-IDF

X_test_tfidf = vectorizer.transform(X_test)

# 5. Prediction

y_pred = model.predict(X_test_tfidf)

# 6. Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\n======")
print("MODEL EVALUATION")
print("======")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

# 7. Classification Report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["ENG", "MIX", "URD"]
    )
)

# 8. Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ENG", "MIX", "URD"]
)

print("\nConfusion Matrix:")
print(cm)

# 9. Display Confusion Matrix

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["ENG", "MIX", "URD"]
)

display.plot()

plt.title("Roman Urdu Language Classification - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("\nConfusion matrix saved as: confusion_matrix.png")

plt.show()
