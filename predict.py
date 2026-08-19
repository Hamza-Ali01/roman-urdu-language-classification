import joblib

# Load Model & Vectorizer

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


print("\n====")
print(" Roman Urdu Language Classifier")
print("====")
print("Model loaded successfully!")
print("Enter a sentence to classify each word.")
print("Type 'exit' to close the program.")
print("====\n")

# Prediction Function

def predict_sentence(sentence):

    words = sentence.split()

    if not words:
        print("Please enter a sentence.")
        return

    print("\n--")
    print("Prediction Result")
    print("--")

    for word in words:

        # Remove simple punctuation
        clean_word = word.strip(".,!?;:")

        if not clean_word:
            continue

        # Convert word into TF-IDF
        word_tfidf = vectorizer.transform([clean_word])

        # Predict label
        prediction = model.predict(word_tfidf)[0]

        print(f"{clean_word:<15} -> {prediction}")

    print("--")

# Main Program

while True:

    sentence = input("\nEnter sentence: ")

    if sentence.lower().strip() == "exit":
        print("\nProgram closed.")
        break

    predict_sentence(sentence)
