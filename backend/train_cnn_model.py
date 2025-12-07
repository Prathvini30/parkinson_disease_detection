import os
import numpy as np
from PIL import Image
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    concatenate,
)
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    """Loads an image, converts it to grayscale, and resizes it."""
    img = Image.open(image_path).convert("L")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(
        img_array, axis=-1
    )  # Add channel dimension for CNN


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
symptoms = [
    "balance",
    "sleep",
    "muscle_stiffness",
    "tremor",
    "speech_difficulty",
]
questionnaire_features = {}

for sample_id, label in labels.items():
    scores = []
    for _ in symptoms:
        if label == 0:  # Healthy
            scores.append(np.random.randint(1, 6))  # Scores 1-5
        else:  # Parkinson's
            scores.append(np.random.randint(6, 11))  # Scores 6-10
    questionnaire_features[sample_id] = np.array(scores)

# --- Feature Combination ---

X_image, X_audio, X_questionnaire, y = [], [], [], []

common_samples = sorted(
    list(
        set(image_features.keys()) \
        & set(audio_features.keys())
        & set(questionnaire_features.keys())
    )
)

for sample_id in common_samples:
    X_image.append(image_features[sample_id])
    X_audio.append(audio_features[sample_id])
    X_questionnaire.append(questionnaire_features[sample_id])
    y.append(labels[sample_id])

X_image = np.array(X_image)
X_audio = np.array(X_audio)
X_questionnaire = np.array(X_questionnaire)
y = np.array(y)

# Normalize audio and questionnaire data
scaler_audio = StandardScaler()
X_audio = scaler_audio.fit_transform(X_audio)

scaler_questionnaire = StandardScaler()
X_questionnaire = scaler_questionnaire.fit_transform(X_questionnaire)


# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"Data prepared. Shapes: {X_image.shape}, {X_audio.shape}, {X_questionnaire.shape}, {y_categorical.shape}")
print(f"Classes found: {le.classes_}")

# --- Model Building (Multimodal CNN) ---


# Image branch (CNN)
image_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), name="image_input")
conv1 = Conv2D(32, (3, 3), activation="relu")(image_input)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation="relu")(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
flatten_img = Flatten()(pool2)
img_dense = Dense(64, activation="relu")(flatten_img)

# Audio branch
audio_input = Input(shape=(X_audio.shape[1],), name="audio_input")
audio_dense = Dense(32, activation="relu")(audio_input)

# Questionnaire branch
questionnaire_input = Input(shape=(X_questionnaire.shape[1],), name="questionnaire_input")
questionnaire_dense = Dense(16, activation="relu")(questionnaire_input)

# Concatenate branches
concatenated = concatenate([img_dense, audio_dense, questionnaire_dense])

# Fully connected layers
dense1 = Dense(128, activation="relu")(concatenated)
output = Dense(y_categorical.shape[1], activation="softmax")(dense1)

# Create the model
model = Model(
    inputs=[image_input, audio_input, questionnaire_input], outputs=output
)

# Compile the model
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

# --- Model Training ---

# Split data
(
    X_img_train, X_img_test,
    X_aud_train, X_aud_test,
    X_ques_train, X_ques_test,
    y_train, y_test,
) = train_test_split(
    X_image, X_audio, X_questionnaire, y_categorical,
    test_size=0.2, random_state=42, stratify=y_categorical
)


print("Training Multimodal CNN...")
history = model.fit(
    [X_img_train, X_aud_train, X_ques_train],
    y_train,
    validation_data=([X_img_test, X_aud_test, X_ques_test], y_test),
    epochs=20,  # Increased epochs for better training
    batch_size=8,
    verbose=1,
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ... (rest of the imports)
# ... (rest of the file before evaluation)

# --- Evaluation ---
y_pred_proba = model.predict(
    [X_img_test, X_aud_test, X_ques_test]
)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred, average='weighted')
recall = recall_score(y_test_labels, y_pred, average='weighted')
f1 = f1_score(y_test_labels, y_pred, average='weighted')

print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")
print(f"Model Precision on Test Set: {precision * 100:.2f}%")
print(f"Model Recall on Test Set: {recall * 100:.2f}%")
print(f"Model F1-score on Test Set: {f1 * 100:.2f}%")

# --- Save the Model, Scalers, and Label Encoder ---
model.save(os.path.join(output_dir, "parkinson_cnn_model.h5"))
joblib.dump(scaler_audio, os.path.join(output_dir, "scaler_audio.joblib"))
joblib.dump(scaler_questionnaire, os.path.join(output_dir, "scaler_questionnaire.joblib"))
joblib.dump(le, os.path.join(output_dir, "label_encoder_cnn.joblib"))


print(f"Model saved to {os.path.join(output_dir, 'parkinson_cnn_model.h5')}")
print(f"Scalers and label encoder saved in {output_dir}")
