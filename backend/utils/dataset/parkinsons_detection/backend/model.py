import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.utils.image_processing import X_train, X_test, y_train, y_test  # Import preprocessed data

# ✅ Check GPU Availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# ✅ Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),  # First Conv Layer
    MaxPooling2D((2, 2)),  # First Pooling Layer
    
    Conv2D(64, (3, 3), activation='relu'),  # Second Conv Layer
    MaxPooling2D((2, 2)),  # Second Pooling Layer
    
    Conv2D(128, (3, 3), activation='relu'),  # Third Conv Layer
    MaxPooling2D((2, 2)),  # Third Pooling Layer
    
    Flatten(),  # Flatten Layer
    Dense(128, activation='relu'),  # Fully Connected Layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(2, activation='softmax')  # Output Layer (2 Classes: Parkinson's & Healthy)
])

# ✅ Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Print Model Summary
model.summary()

# ✅ Train the Model
history = model.fit(
    X_train, y_train,
    epochs=20,  # Adjust based on results
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# ✅ Save the Model
model.save(os.path.join("backend", "parkinson_model.h5"))
print("✅ Model training complete and saved as 'parkinson_model.h5'!")
