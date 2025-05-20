from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
cv = joblib.load("count_vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_message = ""
    if request.method == "POST":
        user_message = request.form["message"]
        vector = cv.transform([user_message])
        result = model.predict(vector)[0]
        prediction = " SPAM" if result else " HAM"
    return render_template("index.html", prediction=prediction, message=user_message)

if __name__ == "__main__":
    app.run(debug=True)
