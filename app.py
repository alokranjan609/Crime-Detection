import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('PATH_TO_YOUR_SAVED_MODEL')

# Set parameters
IMG_SIZE = 224  # Image size for resizing
LABELS = [
    'Abuse',
    'Arrest',
    'Arson',
    'Assault',
    'Burglary',
    'Explosion',
    'Fighting',
    'Normal Videos',
    'RoadAccidents',
    'Robbery',
    'Shooting',
    'Shoplifting',
    'Stealing',
    'Vandalism'
]

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Function to display the class label on the frame
def display_label(frame, label, confidence):
    text = f"{label}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

# Open video capture
cap = cv2.VideoCapture('path_to_your_video.mp4')  # Replace with the path to your video file or 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Inference
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Get the class label
    label = LABELS[predicted_class]
    
    # Display the label on the frame
    frame = display_label(frame, label, confidence)
    
    # Show the frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
