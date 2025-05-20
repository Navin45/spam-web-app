import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
cv = joblib.load("count_vectorizer.pkl")

# Test with custom messages
custom_messages = [
    "Congratulations! You have won a free ticket to Bahamas!",
    "Hi Navin, can you call me back when you're free?",
    "Claim your free gift card now!!!"
]

# Transform and predict
X_custom = cv.transform(custom_messages)
predictions = model.predict(X_custom)

# Show results
for message, label in zip(custom_messages, predictions):
    result = "SPAM" if label else "HAM"
    print(f"[{result}] ➜ {message}")
