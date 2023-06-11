import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from itertools import permutations

# Specify the path to the dataset directory
dataset_path = "D:\Programming\Hackathon\dataset"

# Generate all permutations of the characters in "UNICUS"
permutations_list = list(permutations("UNICUS"))

# Load the dataset
def load_dataset():
    images = []
    labels = []
    characters = ["U", "N", "I", "C", "U_2", "S"]
    target_labels = ["".join(permutation) for permutation in permutations_list]

    for character, target_label in zip(characters, target_labels):
        character_path = os.path.join(dataset_path, character)
        if os.path.isdir(character_path):
            for image_file in os.listdir(character_path):
                image_path = os.path.join(character_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue
                image = cv2.resize(image, (32, 32))
                images.append(image)
                labels.append(target_label)

    return images, labels

# Preprocess the dataset
def preprocess_dataset(images, labels):
    images = np.array(images)
    images = images.astype('float32') / 255.0

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels

# Load and preprocess the dataset
images, labels = load_dataset()
images, labels = preprocess_dataset(images, labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(permutations_list), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Testing loss: {loss:.4f}')
print(f'Testing accuracy: {accuracy:.4f}')

# Save the trained model
model.save('trained_model8.h5')