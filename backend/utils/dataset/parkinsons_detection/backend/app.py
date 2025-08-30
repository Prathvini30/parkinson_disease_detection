from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load the trained CNN model
MODEL_PATH = "parkinson_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found! Train and save the model first.")

model = load_model(MODEL_PATH)

# Define image size (must match model input)
IMG_SIZE = (128, 128)

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend integration (Flutter/React)

# Image Preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(IMG_SIZE)  # Resize to match CNN input
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, 128, 128, 1)  # Reshape for model input
    return img

# Home Route
@app.route("/")
def home():
    return "✅ Parkinson's Detection API is running!"

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = "uploaded_image.png"
    file.save(filename)

    # Preprocess the uploaded image
    img = preprocess_image(filename)

    # Get prediction from model
    prediction = model.predict(img)
    label = "Parkinson's Detected" if np.argmax(prediction) == 1 else "Healthy"
    accuracy = float(np.max(prediction) * 100)

    # Delete temporary image file
    os.remove(filename)

    return jsonify({"prediction": label, "accuracy": accuracy})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
