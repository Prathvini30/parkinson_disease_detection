from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from PIL import Image
import librosa
import os
import json
import tensorflow as tf

# --- Define Constants ---
IMG_SIZE = (128, 128)
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load Model, Scalers, and Encoder ---
MODEL_PATH = os.path.join(script_dir, "parkinson_cnn_model.h5")
SCALER_AUDIO_PATH = os.path.join(script_dir, "scaler_audio.joblib")
SCALER_QUES_PATH = os.path.join(script_dir, "scaler_questionnaire.joblib")
ENCODER_PATH = os.path.join(script_dir, "label_encoder_cnn.joblib")

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_AUDIO_PATH, SCALER_QUES_PATH, ENCODER_PATH]):
    raise FileNotFoundError("A required model, scaler, or encoder file is missing! Please train the CNN model first.")

model = tf.keras.models.load_model(MODEL_PATH)
scaler_audio = joblib.load(SCALER_AUDIO_PATH)
scaler_questionnaire = joblib.load(SCALER_QUES_PATH)
label_encoder = joblib.load(ENCODER_PATH)

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend integration

# --- Preprocessing Functions ---

def preprocess_image_for_cnn(image_path):
    """Loads and preprocesses an image for the CNN model."""
    img = Image.open(image_path).convert("L")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=[0, -1])  # (1, 128, 128, 1)

def preprocess_audio(audio_path):
    """Loads an audio file and extracts the mean of MFCCs."""
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1).reshape(1, -1) # (1, 13)

# --- Routes ---

@app.route("/")
def home():
    return "✅ Parkinson's Detection API (CNN Model) is running!"

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

    expected_symptoms = ['balance', 'sleep', 'muscle_stiffness', 'tremor', 'speech_difficulty']
    ques_feat_list = []
    for symptom in expected_symptoms:
        if symptom not in questionnaire_scores:
            return jsonify({"error": f"Missing questionnaire score for {symptom}"}), 400
        ques_feat_list.append(questionnaire_scores[symptom])
    
    ques_feat = np.array(ques_feat_list).reshape(1, -1) # (1, 5)

    # Save temporary files
    image_filename = "temp_image.png"
    audio_filename = "temp_audio.webm"
    image_file.save(image_filename)
    audio_file.save(audio_filename)

    try:
        # Preprocess all inputs
        image_input = preprocess_image_for_cnn(image_filename)
        audio_input = scaler_audio.transform(preprocess_audio(audio_filename))
        questionnaire_input = scaler_questionnaire.transform(ques_feat)

        # Get prediction from the multimodal model
        prediction_proba = model.predict([image_input, audio_input, questionnaire_input])
        
        # Decode prediction
        prediction_encoded = np.argmax(prediction_proba, axis=1)
        class_label_num = label_encoder.inverse_transform(prediction_encoded)[0]
        class_label = "Healthy" if class_label_num == 0 else "Parkinson's Detected"
        
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
