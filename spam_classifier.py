import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']  # Rename columns

# Encode labels: ham = 0, spam = 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize text
cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label_num']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


sample = ["Free entry in 2 a wkly comp to win FA Cup", "Hey, can we meet today?"]
sample_vector = cv.transform(sample)
predictions = model.predict(sample_vector)

for msg, pred in zip(sample, predictions):
    print(f"{'SPAM' if pred else 'HAM'} ➜ {msg}")


# Set plot style
sns.set(style='whitegrid')

# Plot spam vs ham counts
sns.countplot(data=df, x='label')
plt.title("Spam vs Ham Message Count")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# Pie chart
df['label'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Ham', 'Spam'], colors=['lightgreen', 'lightcoral'])
plt.title("Distribution of Spam and Ham Messages")
plt.ylabel("")  # Hide y-axis label
plt.show()


# Save model and vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(cv, "count_vectorizer.pkl")

print(" Model and vectorizer saved successfully.")

