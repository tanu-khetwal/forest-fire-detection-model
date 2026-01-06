import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import threading
import time  

# --- Configuration ---
VIDEO_SOURCE = 0  # <-- This is your webcam
MODEL_PATH = 'fire_model.h5' # <-- !! THE FIX !! This is your new, "smart" model
IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- !! LOGIC CORRECTED FROM train.py !! ---
FIRE_PREDICTION_THRESHOLD = 0.15  # <-- Tuned threshold
# ----------------------------------------

# --- Global Variables ---
output_frame = None
lock = threading.Lock()
fire_alert = False  # Global variable to store fire detection status

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Model Loading ---
try:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'fire_model.h5' exists and 'train.py' has been run.")
    exit()

# --- Video Capture ---
print(f"Attempting to open video source: {VIDEO_SOURCE}")
video_capture = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
if not video_capture.isOpened():
    print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
    print("Is the camera connected or the file path correct?")
    exit()
else:
    print("Video source opened successfully.")

def preprocess_frame(frame):
    """
    Prepares a single video frame for model prediction.
    """
    try:
        img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img_rgb / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        #print(f"Error in preprocess_frame: {e}")
        return None

def detect_fire():
    """
    The main detection loop that runs in a separate thread.
    Reads frames, predicts, and updates global variables.
    """
    global output_frame, lock, fire_alert, video_capture # <-- Added global video_capture
    last_print_time = time.time()
    last_error_time = 0

    while True:
        # --- !! NEW SELF-HEALING LOGIC !! ---
        if not video_capture.isOpened():
            if time.time() - last_error_time > 5.0: # Try to re-open every 5s
                print("Webcam is not open. Trying to re-open...")
                video_capture.release()
                video_capture = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
                if not video_capture.isOpened():
                    print("Failed to re-open webcam. Retrying in 5s.")
                    last_error_time = time.time()
                else:
                    print("Webcam re-opened successfully.")
            time.sleep(1) # Wait a bit before retrying
            continue

        ret, frame = video_capture.read()
        
        if not ret or frame is None: # <-- Added check for None frame
            # Frame read failed. This is the user's error.
            if time.time() - last_error_time > 5.0: # Don't spam
                print("Video stream ended or frame read failed. Re-initializing camera...")
                video_capture.release()
                video_capture = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
                if not video_capture.isOpened():
                    print("Failed to re-open webcam after frame fail. Retrying in 5s.")
                    last_error_time = time.time()
                else:
                    print("Webcam re-initialized successfully.")
            
            time.sleep(1) # Wait a bit before retrying
            continue
        
        # --- !! NEW FRAME SANITY CHECK !! ---
        # If the frame has no width or height, it's corrupt. Skip it.
        if frame.shape[0] < 1 or frame.shape[1] < 1:
            print("Received corrupt frame (empty shape). Skipping.")
            continue
        # --- !! END OF NEW CHECK !! ---
        
        processed_frame = preprocess_frame(frame)
        if processed_frame is None:
            continue
        
        try:
            # prediction = LOW (fire), HIGH (not-fire)
            prediction = model.predict(processed_frame, verbose=0)[0][0]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

        # --- !! DEBUG PRINT !! ---
        current_time = time.time()
        if current_time - last_print_time > 1.0: # Print once per second
            print(f"Raw Prediction: {prediction:.4f}")
            last_print_time = current_time
        # -----------------------------

        # --- !! LOGIC FLIPPED BACK !! ---
        
        # 1. CHECK: 'fire' is class 0, so a LOW prediction means fire.
        is_fire = prediction < FIRE_PREDICTION_THRESHOLD
        
        # 2. CALCULATE CONFIDENCE: (1.0 - prediction)
        # (e.g., 0.1 prediction = 90% confident it's fire)
        fire_confidence_percent = (1.0 - prediction) * 100
        
        # ------------------------------------

        fire_alert = bool(is_fire) # Use bool() for JSON fix

        # --- Draw visuals on the frame ---
        if is_fire:
            label = "FIRE DETECTED!"
            prob_text = f"Fire Confidence: {fire_confidence_percent:.2f}%"
            color = (0, 0, 255)  # Red
            
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
            cv2.putText(frame, prob_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        
        else:
            label = "System Safe"
            prob_text = f"Fire Confidence: {fire_confidence_percent:.2f}%"
            color = (0, 255, 0)  # Green
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(frame, prob_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
        with lock:
            output_frame = frame.copy()
            
    # This code should no longer be reachable, but we leave it for safety.
    video_capture.release()
    print("Detection loop has fully stopped.")

def generate_video_stream():
    """
    Generator function to yield video frames for the web stream.
    """
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                continue
            
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML dashboard."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provides the MJPEG video stream."""
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """API endpoint to provide the fire status to the frontend."""
    global fire_alert
    return jsonify(fire_detected=bool(fire_alert))


# --- Main Entry Point ---
if __name__ == '__main__':
    # Start the background thread for fire detection
    detection_thread = threading.Thread(target=detect_fire, daemon=True)
    detection_thread.start()
    
    # Start the Flask web server
    print("Starting Flask server... Access at http://127.0.0.1:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)