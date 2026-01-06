import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10

# --- !! UPDATED FILE PATHS !! ---
# These paths now point to the folders you described.
TRAIN_DIR = 'dataset/Forest Fire Dataset/Training'
# VALID_DIR has been removed
MODEL_SAVE_PATH = 'fire_model.h5'
# ----------------------------------

# --- Improved Directory Check ---
# This check is more specific and checks for the subfolders.
train_fire_dir = os.path.join(TRAIN_DIR, 'fire')
train_nofire_dir = os.path.join(TRAIN_DIR, 'nofire')

if not (os.path.exists(train_fire_dir) and os.path.exists(train_nofire_dir)):
    print("="*50)
    print(f"ERROR: Training directory '{TRAIN_DIR}' is missing 'fire' or 'nofire' subfolders.")
    print("Please make sure your training data is organized correctly.")
    print("="*50)
    exit()
# --- End of Check ---


# --- 1. Build the CNN Model ---
# We use a simple, lightweight CNN to ensure it runs on a CPU.
def build_model():
    model = Sequential([
        # Layer 1: Convolution + Pooling
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(2, 2),

        # Layer 2: Convolution + Pooling
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Layer 3: Convolution + Pooling
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Layer 4: Flatten and Dense
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout for regularization to prevent overfitting

        # Output Layer:
        # We use 'sigmoid' because this is a binary (2-class) classification
        # The output will be a single value between 0 (class 0) and 1 (class 1)
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# --- 2. Prepare Data (Data Augmentation) ---
# We use ImageDataGenerator to automatically label data based on
# folder structure and apply random augmentations (like zoom, flip)
# to make our model more robust to different image variations.

# Rescale all images by 1./255
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# --- Validation generator has been removed ---

# Flow training images in batches
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Critical for binary_crossentropy
)

print(f"Class indices found: {train_generator.class_indices}")
# This will print: {'fire': 0, 'nofire': 1}
# This confirms 'fire' is class 0, and 'nofire' is class 1.
# Your app.py logic MUST match this.

# --- 3. Train the Model ---
print("Starting model training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS
)
# --- !! END FIX !! ---

# --- 4. Save the Model ---
print(f"Training complete. Saving model to {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")