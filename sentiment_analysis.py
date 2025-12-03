# Sentiment Analysis Project using NLP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "text": [
        "I love this product, it is amazing!",
        "Worst experience ever, I hate it.",
        "It was okay, not too bad.",
        "Absolutely fantastic service!",
        "This is terrible and disappointing.",
        "I feel happy using this app.",
        "Not good, not bad, just average.",
        "Best purchase I made this year!",
        "I will never buy this again.",
        "Pretty decent for the price."
    ],
    "label": [
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Positive",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral"
    ]
}
df = pd.DataFrame(data)
print("✓ Sample dataset created")
print(df)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\n✓ Text converted to TF-IDF vectors")
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
print("✓ Model training completed")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")
print("\n===== Sentiment Prediction =====")
user_text = input("Enter a review: ")

user_text_tfidf = vectorizer.transform([user_text])
prediction = model.predict(user_text_tfidf)

print("\nPredicted Sentiment:", prediction[0])

