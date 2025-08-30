import os
import numpy as np
from PIL import Image
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spiral_dir = os.path.join(base_dir, "spiral_images")
voice_dir = os.path.join(base_dir, "voice_samples")
output_dir = os.path.join(base_dir, "backend")

# Image settings
IMG_SIZE = (128, 128)

# --- Feature Extraction ---

def extract_image_features(image_path):
    """Loads an image, converts it to grayscale, resizes, and flattens it."""
    img = Image.open(image_path).convert("L")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return img_array.flatten()

def extract_audio_features(audio_path):
    """Loads an audio file and extracts the mean of MFCCs."""
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# --- Data Loading and Preparation ---

image_features = {}
audio_features = {}
labels = {}

# Process spiral images
for filename in os.listdir(spiral_dir):
    if filename.endswith(".png"):
        sample_id = os.path.splitext(filename)[0]
        label = 0 if "healthy" in filename else 1
        
        image_path = os.path.join(spiral_dir, filename)
        image_features[sample_id] = extract_image_features(image_path)
        labels[sample_id] = label

# Process voice samples
for filename in os.listdir(voice_dir):
    if filename.endswith(".wav"):
        sample_id = os.path.splitext(filename)[0]
        
        audio_path = os.path.join(voice_dir, filename)
        audio_features[sample_id] = extract_audio_features(audio_path)
        # The label is already set from the image processing step

# --- Synthetic Questionnaire Data Generation ---
# Define symptoms and generate synthetic scores based on label
symptoms = ['balance', 'sleep', 'muscle_stiffness', 'tremor', 'speech_difficulty']
questionnaire_features = {}

for sample_id, label in labels.items():
    scores = []
    for _ in symptoms:
        if label == 0: # Healthy
            scores.append(np.random.randint(1, 6)) # Scores 1-5
        else: # Parkinson's
            scores.append(np.random.randint(6, 11)) # Scores 6-10
    questionnaire_features[sample_id] = np.array(scores)

# --- Feature Combination ---

combined_features = []
final_labels = []

# Ensure we only use samples that have image, audio, and questionnaire data
common_samples = sorted(list(set(image_features.keys()) & set(audio_features.keys()) & set(questionnaire_features.keys())))

for sample_id in common_samples:
    img_feat = image_features[sample_id]
    aud_feat = audio_features[sample_id]
    ques_feat = questionnaire_features[sample_id]
    
    # Concatenate all features
    combined_feat = np.concatenate([img_feat, aud_feat, ques_feat])
    
    combined_features.append(combined_feat)
    final_labels.append(labels[sample_id])

X = np.array(combined_features)
y = np.array(final_labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Data prepared. Shape of X: {X.shape}, Shape of y: {y_encoded.shape}")
print(f"Classes found: {le.classes_}")


# --- Model Training ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Initialize and train the RandomForestClassifier
print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# --- Save the Model and Label Encoder ---

model_path = os.path.join(output_dir, "parkinson_combined_model.joblib")
encoder_path = os.path.join(output_dir, "label_encoder.joblib")

joblib.dump(model, model_path)
joblib.dump(le, encoder_path)

print(f"Model saved to {model_path}")
print(f"Label encoder saved to {encoder_path}")