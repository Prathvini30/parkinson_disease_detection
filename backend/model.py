import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.utils.image_processing import X_train, X_test, y_train, y_test  # Preprocessed data

# ✅ Check GPU Availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# ✅ Define a smaller, faster CNN Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
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
    epochs=5,          # small dataset, fewer epochs
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# ✅ Save the Model
model.save(os.path.join("backend", "parkinson_model.keras"))
print("✅ Model training complete and saved as 'parkinson_model.keras'!")
