import tensorflow as tf
import cv2
import pytesseract as tess
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('trained_model9.h5')

# Set the path to Tesseract executable
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to preprocess the image for the model
def preprocess_image(image):
    # Preprocess the image (resize, normalize, etc.)
    image = cv2.resize(image, (32, 32))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype('float32') / 255.0
    gray_image = tf.expand_dims(gray_image, axis=-1)
    gray_image = tf.expand_dims(gray_image, axis=0)
    return gray_image

# Function to recognize text using Tesseract
def recognize_text(image):
    text = tess.image_to_string(image)
    return text

# Read the image
image = cv2.imread('test.png')

# Preprocess the image for the model
processed_image = preprocess_image(image)

# Recognize text using Tesseract
extracted_text = recognize_text(image)

# Feed the preprocessed image to the model
prediction = model.predict(processed_image)

# Get the predicted text
predicted_class = tf.argmax(prediction, axis=1)
text = predicted_class.numpy()[0]

# Display the recognized text
print("Extracted Text:", extracted_text)
print("Predicted Text:", text)
