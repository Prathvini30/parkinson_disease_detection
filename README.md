🧠 Parkinson's Disease Detection – Multimodal Deep Learning System

This project is a full-stack web application that predicts the likelihood of Parkinson’s Disease using three different inputs: spiral drawings, voice recordings, and symptom scores. It uses a multimodal deep learning model to give a combined prediction with confidence.

🚀 Features
🌀 1. Spiral Drawing Analysis

Users upload a hand-drawn spiral image. A CNN model analyzes shape irregularities related to motor dysfunction.

🎤 2. Voice Analysis

Users record a short audio sample. Using Librosa, MFCC features are extracted to detect vocal tremor and instability.

📝 3. Symptom Questionnaire

A lightweight symptom form captures self-reported scores (tremor, stiffness, balance, sleep, speech).

🤖 4. Multimodal Prediction

A deep learning model merges image, audio, and symptom features to classify between:

Healthy

Parkinson’s Detected

Returns both prediction + confidence score.

🧩 Technology Stack
Backend

Python

Flask

TensorFlow / Keras

Scikit-learn

Librosa

NumPy / Pandas

Frontend

React.js

JavaScript

HTML / CSS

🧠 Final Model – Multimodal CNN

The final architecture merges three branches:

🔹 Image Branch (CNN)

Extracts spatial features from spiral images.

🔹 Audio Branch

Dense network trained on MFCC features from Librosa.

🔹 Questionnaire Branch

Fully connected network processing numerical symptom scores.

The three outputs are concatenated → passed to final Dense layers → binary prediction.

📊 Model Performance

On the current test set:

Metric	Score
Accuracy	100%
Precision	100%
Recall	100%
F1-Score	100%

⚠️ Note: These results are dataset-specific. Real-world accuracy requires a larger and clinically validated dataset.

🛠️ Project Setup
✔️ Prerequisites

Python 3.8+

Node.js + npm

🖥️ Backend Setup (Flask)
git clone <your-repository-url>
cd parkinson_detection/backend

Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# OR
source .venv/bin/activate  # macOS/Linux

Install Dependencies
pip install -r requirements.txt

🌐 Frontend Setup (React)
cd ../frontend
npm install

▶️ How to Run the Application
1️⃣ (Optional) Retrain the Model
python train_cnn_model.py

2️⃣ Start the Backend
python app.py


Backend will run at:

http://127.0.0.1:5000

3️⃣ Start the Frontend
npm start


Frontend opens at:

http://localhost:3000
