import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'fire_model.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224
CONFIDENCE_THRESHOLD = 0.5 # Adjust if needed based on your class indices

# --- Load Model ---
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded!")
except IOError:
    print(f"ERROR: Could not find {MODEL_PATH}")
    exit()

# --- Start Webcam ---
# 0 is usually the built-in webcam. Try 1 if you have an external one.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 2. Preprocess frame for the model
    # Resize to 224x224 matches your training input
    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    # Convert to float and normalize (0-1 range), matches rescale=1./255
    img_array = np.array(resized_frame, dtype="float32") / 255.0
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Make Prediction
    prediction = model.predict(img_batch, verbose=0)
    probability = prediction[0][0]

    # 4. Interpret Result & Display on Screen
    # ASSUMPTION: Class 0 = 'fire', Class 1 = 'no_fire'
    # This means low probability (near 0) is FIRE.
    if probability < CONFIDENCE_THRESHOLD:
        label = "FIRE DETECTED! "
        color = (0, 0, 255) # Red in BGR
        confidence_val = (1 - probability) * 100
    else:
        label = "Normal"
        color = (0, 255, 0) # Green in BGR
        confidence_val = probability * 100

    text = f"{label} ({confidence_val:.1f}%)"

    # Draw label on the original frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Draw a rectangle around the screen border just for visual effect if fire detected
    if label.startswith("FIRE"):
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), color, 10)

    # 5. Show the resulting frame
    cv2.imshow('Fire Detection - Webcam', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()