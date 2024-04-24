from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("best_diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get form data
        preg = int(request.form["pregnancies"])
        glucose = int(request.form["glucose"])
        bp = int(request.form["blood_pressure"])
        skin = int(request.form["skin_thickness"])
        insulin = int(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])
        age = int(request.form["age"])
        
        # Make prediction
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(data)
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        
        return render_template("result.html", result=result)
# About Us route
@app.route("/about")
def about():
    return render_template("about.html")

# Contact Us route
@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
