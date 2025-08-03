from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
target_names = joblib.load("target_names.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        features = [float(request.form[f]) for f in ['sl', 'sw', 'pl', 'pw']]
        prediction = model.predict([features])[0]
        result = target_names[prediction]
        return render_template("index.html", result=result)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
