import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Get the absolute path of the utils/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define dataset paths inside utils/
dataset_dir = os.path.join(BASE_DIR, "dataset")
parkinson_dir = os.path.join(dataset_dir, "parkinson")
healthy_dir = os.path.join(dataset_dir, "healthy")

# Function to Load Images
def load_images_from_folder(folder, label):
    if not os.path.exists(folder):
        print(f"⚠️ Error: Folder {folder} does not exist!")
        return [], []
    
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize image to 128x128
            images.append(img)
            labels.append(label)
    
    return images, labels

# ✅ Check if dataset folders exist
print(f"Checking dataset folder: {dataset_dir} Exists: {os.path.exists(dataset_dir)}")
print(f"Checking Parkinson folder: {parkinson_dir} Exists: {os.path.exists(parkinson_dir)}")
print(f"Checking Healthy folder: {healthy_dir} Exists: {os.path.exists(healthy_dir)}")

# Load Images
parkinson_images, parkinson_labels = load_images_from_folder(parkinson_dir, 1)
healthy_images, healthy_labels = load_images_from_folder(healthy_dir, 0)

# Combine Both Classes
X = np.array(parkinson_images + healthy_images)
y = np.array(parkinson_labels + healthy_labels)

# ✅ Normalize pixel values (0 to 1)
X = X / 255.0  

# ✅ Reshape X to (num_samples, 128, 128, 1) for CNN input
X = X.reshape(-1, 128, 128, 1)

# ✅ Convert labels to categorical (one-hot encoding)
y = to_categorical(y, 2)

# ✅ Split Dataset into Training and Testing (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Check Dataset Balance
unique, counts = np.unique(y.argmax(axis=1), return_counts=True)
print(f"Dataset Balance: {dict(zip(unique, counts))}")

# ✅ Compute Class Weights (useful if dataset is imbalanced)
class_weights = compute_class_weight("balanced", classes=np.unique(y.argmax(axis=1)), y=y.argmax(axis=1))
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Computed Class Weights: {class_weight_dict}")

# ✅ Debugging Output
print("Final dataset prepared!")
print("Shape of X_train:", X_train.shape)  # Expected: (80% of samples, 128, 128, 1)
print("Shape of X_test:", X_test.shape)  # Expected: (20% of samples, 128, 128, 1)
print("Shape of y_train:", y_train.shape)  # Expected: (80% of samples, 2)
print("Shape of y_test:", y_test.shape)  # Expected: (20% of samples, 2)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ✅ STEP 4: Visualize Some Images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i].reshape(128, 128), cmap='gray')
    plt.title("Parkinson" if np.argmax(y_train[i]) == 1 else "Healthy")
    plt.axis('off')
plt.show()
