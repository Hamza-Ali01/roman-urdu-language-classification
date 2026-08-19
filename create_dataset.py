import pandas as pd

# 10 sample Roman Urdu / English sentences
sentences = [
    "please laptop open karo",
    "kal meri online class hai",
    "tum file download karo",
    "main project upload kar raha hoon",
    "mujhe new software install karna hai",
    "teacher ne homework diya hai",
    "meeting ka time kya hai",
    "mera computer slow chal raha hai",
    "main school assignment complete kar raha hoon",
    "please mera project check karo"
]

# Roman Urdu words
urdu_words = {
    "main", "mujhe", "ja", "raha", "rahi", "hoon",
    "karna", "kar", "karo", "tum", "meri", "mera",
    "ne", "diya", "ka", "ki", "kya", "hai",
    "chal", "slow"
}

# English words
english_words = {
    "school", "assignment", "complete", "please",
    "laptop", "open", "kal", "online", "class",
    "file", "download", "project", "upload", "new",
    "software", "install", "teacher", "homework",
    "meeting", "time", "computer", "check"
}


def get_label(word):
    word = word.lower()

    if word in urdu_words:
        return "URD"
    elif word in english_words:
        return "ENG"
    else:
        return "MIX"


data = []

for sentence_id, sentence in enumerate(sentences, start=1):

    words = sentence.split()

    for word in words:
        label = get_label(word)

        data.append({
            "Sentence_ID": sentence_id,
            "Sentence": sentence,
            "Word": word,
            "Label": label
        })


# DataFrame create karo
df = pd.DataFrame(data)

# CSV file save karo
df.to_csv("dataset.csv", index=False, encoding="utf-8-sig")

print("Dataset created successfully!")
print(f"Total sentences: {len(sentences)}")
print(f"Total words: {len(df)}")

print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nFirst 20 rows:")
print(df.head(20))