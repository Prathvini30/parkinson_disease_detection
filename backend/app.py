from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from PIL import Image
import librosa
import os
import json

# Define image size
IMG_SIZE = (128, 128)

# --- Load Model and Encoder ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "parkinson_combined_model.joblib")
ENCODER_PATH = os.path.join(script_dir, "label_encoder.joblib")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("Model or encoder file not found! Train and save the model first.")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend integration

# --- Preprocessing Functions ---

def preprocess_image(image_path):
    """Loads, preprocesses, and flattens an image."""
    img = Image.open(image_path).convert("L")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return img_array.flatten()

def preprocess_audio(audio_path):
    """Loads an audio file and extracts the mean of MFCCs."""
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# --- Routes ---

@app.route("/")
def home():
    return "✅ Parkinson's Detection API (Combined Model) is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "audio" not in request.files or "questionnaire_scores" not in request.form:
        return jsonify({"error": "Image, audio, and questionnaire scores are required"}), 400

    image_file = request.files["image"]
    audio_file = request.files["audio"]
    
    try:
        questionnaire_scores = json.loads(request.form["questionnaire_scores"])
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON for questionnaire_scores"}), 400

    if image_file.filename == '' or audio_file.filename == '':
        return jsonify({"error": "No selected file(s)"}), 400

    # Define expected symptoms (must match training script)
    expected_symptoms = ['balance', 'sleep', 'muscle_stiffness', 'tremor', 'speech_difficulty']
    ques_feat = []
    for symptom in expected_symptoms:
        if symptom not in questionnaire_scores:
            return jsonify({"error": f"Missing questionnaire score for {symptom}"}), 400
        ques_feat.append(questionnaire_scores[symptom])
    ques_feat = np.array(ques_feat)

    # Save temporary files
    image_filename = "temp_image.png"
    audio_filename = "temp_audio.webm"
    image_file.save(image_filename)
    audio_file.save(audio_filename)

    try:
        # Preprocess inputs
        image_features = preprocess_image(image_filename)
        audio_features = preprocess_audio(audio_filename)

        # Combine features
        combined_features = np.concatenate([image_features, audio_features, ques_feat])
        combined_features = combined_features.reshape(1, -1) # Reshape for single prediction

        # Get prediction
        prediction_encoded = model.predict(combined_features)
        prediction_proba = model.predict_proba(combined_features)
        
        # Decode prediction
        label = label_encoder.inverse_transform(prediction_encoded)[0]
        # The label is 0 for healthy and 1 for parkinson, so we can use the label directly
        class_label = "Healthy" if label == 0 else "Parkinson's Detected"
        
        # Get confidence score
        confidence = float(np.max(prediction_proba) * 100)

    finally:
        # Clean up temporary files
        if os.path.exists(image_filename):
            os.remove(image_filename)
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

    return jsonify({"prediction": class_label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)