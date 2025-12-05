from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# -------------------------
# Load saved model & label encoder
# -------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_svm.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare input for model
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        pred_class = model.predict(input_data)[0]
        pred_species = label_encoder.inverse_transform([pred_class])[0]
        confidence = np.max(model.predict_proba(input_data)) * 100

        # Conditional display
        data = 0 if pred_class == 0 else 1

        return render_template('after.html', 
                               data=data, 
                               species=pred_species.capitalize(), 
                               confidence=f"{confidence:.2f}%")
    except Exception as e:
        return redirect(url_for('home'))

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
