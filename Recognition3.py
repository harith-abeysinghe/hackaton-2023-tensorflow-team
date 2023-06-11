import cv2
import pytesseract
import tensorflow as tf
import numpy as np

# Path to Tesseract executable (change this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the trained machine learning model
model = tf.keras.models.load_model('trained_model8.h5')

# Define the preprocessing function for the machine learning model
def preprocess_frame(frame):
    # Preprocess the frame (resize, normalize, etc.)
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)
    processed_frame = cv2.threshold(processed_frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    processed_frame = cv2.resize(processed_frame, (32, 32))
    processed_frame = processed_frame.astype('float32') / 255.0
    return processed_frame

# Define the decoding function for the machine learning model output
def decode_text(predicted_text):
    # Decode the predicted text (specific to your model's output format)
    # Implement your own decoding logic here based on your model's output format
    return predicted_text

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Apply color-based segmentation to isolate the iPad
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Preprocess the frame for text extraction (optional)
    gray = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # Perform text extraction using pytesseract (optional)
    text = pytesseract.image_to_string(gray)

    # Preprocess the frame for the machine learning model
    processed_frame = preprocess_frame(segmented_frame)

    # Add an extra dimension to match the input shape of the model
    processed_frame = np.expand_dims(processed_frame, axis=0)

    # Perform text recognition using the loaded machine learning model
    predicted_text = model.predict(processed_frame)

    # Decode the predicted text
    decoded_text = decode_text(predicted_text)

    # Draw the recognized text on the frame
    cv2.putText(frame, str(decoded_text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame in a window
    cv2.imshow('Webcam', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()