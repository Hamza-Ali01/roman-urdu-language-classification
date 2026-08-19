import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset

df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))

# 2. Input and Target

X = df["Word"]
y = df["Label"]

# 3. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. TF-IDF

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF feature extraction completed!")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

# 5. Train Model

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# 6. Evaluation

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n====")
print("MODEL RESULTS")
print("====")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Model

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n====")
print("FILES SAVED")
print("====")

print("Model saved as: model.pkl")
print("Vectorizer saved as: vectorizer.pkl")

# 8. Prediction Function

def predict_word(word):

    word_tfidf = vectorizer.transform([word])

    prediction = model.predict(word_tfidf)

    return prediction[0]

# 9. Test Predictions

test_words = [
    "main",
    "mujhe",
    "project",
    "upload",
    "kar",
    "please",
    "laptop",
    "software",
    "hoon",
    "teacher"
]

print("\n====")
print("PREDICTION TEST")
print("====")

for word in test_words:

    result = predict_word(word)

    print(f"{word:12} -> {result}")
