import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.models import load_model

'''
1. Fetching data from image
2. Creating a model

'''


# Load the text detection and recognition models.
text_detection_model = load_model("path/to/text_detection_model")
text_recognition_model = load_model("path/to/text_recognition_model")

# Read the image into memory.
image = tf.keras.preprocessing.image.load_img("path/to/image")

# Detect the text in the image.
text_boxes = text_detection_model.predict(image)

# Recognize the text in the image.
recognized_text = []
for text_box in text_boxes:
    recognized_text.append(text_recognition_model.predict(text_box))

# Print the recognized text.
print(recognized_text)