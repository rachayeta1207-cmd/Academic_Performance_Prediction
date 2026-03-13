from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    gpa = float(request.form["gpa"])
    assignments = float(request.form["assignments"])

    features = np.array([[study_hours, attendance, gpa, assignments]])

    prediction = model.predict(features)

    if prediction[0] == 0:
        result = "Low Performance"
    elif prediction[0] == 1:
        result = "Medium Performance"
    else:
        result = "High Performance"

    return render_template(
        "index.html",
        prediction_text="Predicted Result: " + result
    )

if __name__ == "__main__":
    app.run(debug=True)
    