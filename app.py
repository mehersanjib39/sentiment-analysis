import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# 1. Create training dataset
# -----------------------------------

data = {
    "text": [
        "I love this product",
        "This is amazing",
        "I am very happy",
        "Excellent service",
        "The movie was fantastic",
        "I really enjoyed it",
        "This is wonderful",
        "I am satisfied",
        "Very good experience",
        "I like this",
        
        "I hate this product",
        "This is terrible",
        "I am very disappointed",
        "Worst service ever",
        "The movie was horrible",
        "I really disliked it",
        "This is bad",
        "I am unhappy",
        "Very poor experience",
        "I don't like this",

        "It is okay",
        "This is average",
        "Nothing special",
        "It is normal",
        "The product is fine",
        "The movie was okay",
        "I have no opinion",
        "It was neither good nor bad",
        "The service was average",
        "It is acceptable"
    ],

    "sentiment": [
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",

        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",

        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral"
    ]
}

df = pd.DataFrame(data)

# -----------------------------------
# 2. Separate input and output
# -----------------------------------

X = df["text"]
y = df["sentiment"]

# -----------------------------------
# 3. Convert text into numbers
# -----------------------------------

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# -----------------------------------
# 4. Split dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# 5. Train Machine Learning model
# -----------------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------------
# 6. Check accuracy
# -----------------------------------

y_prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)

print("Model Accuracy:", accuracy * 100, "%")

# -----------------------------------
# 7. Take sentence from user
# -----------------------------------

while True:

    sentence = input("\nEnter a sentence (or type 'exit' to stop): ")

    if sentence.lower() == "exit":
        print("Program stopped.")
        break

    # Convert user's sentence into numbers
    sentence_vector = vectorizer.transform([sentence])

    # Predict sentiment
    prediction = model.predict(sentence_vector)

    print("Sentiment:", prediction[0])
