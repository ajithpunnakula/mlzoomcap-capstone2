# %%
import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json


# Load the class indices
with open('./class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}


def load_image_from_url(url, target_size=(150, 150)):
    # Download the image
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    # Convert bytes data to PIL Image
    img = Image.open(BytesIO(response.content))

    # Resize to target size
    img = img.resize(target_size)

    # Convert to numpy array and scale pixel values
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]

    return img_array

def load_image(img_path, target_size=(150, 150)):
    # Load the image and resize it to the target size
    img = image.load_img(img_path, target_size=target_size)
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Expand dims to fit the model input format (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image (same as used in training)
    img_array /= 255.
    return img_array

def predict_dog_breed(img_path, model_path='MobileNetV2_best_model.h5'):
    # Load MobileNetV2 best model weights
    model = load_model(model_path)

    # Load and preprocess the image
    img_array = load_image(img_path)

    # Perform prediction
    predictions = model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class_index]

def predict_dog_breed_from_url(url):
    img_array = load_image_from_url(url)
    # Load MobileNetV2 best model weights
    model = load_model(model_path)
    # Perform prediction
    predictions = model.predict(img_array)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class_index]

# Example usage of the function
# image_path = './Hugsy2.jpg'  # Replace with your image path
image_url = 'https://images.unsplash.com/photo-1507146426996-ef05306b995a'
model_path = './MobileNetV2_best_model.h5'
# predicted_breed = predict_dog_breed(image_path)
# print(f'The predicted dog breed is: {predicted_breed}')
predicted_breed = predict_dog_breed_from_url(image_url)
print(f'The predicted dog breed is: {predicted_breed}')


# %%



