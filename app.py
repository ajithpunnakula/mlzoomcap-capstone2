import json
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from tensorflow.keras.models import load_model

app = FastAPI()

# Pydantic model for request body
class ImageUrl(BaseModel):
    url: str

# Paths to model and class index files
model_path = './MobileNetV2_best_model.h5'
class_indices_path = './class_indices.json'

# Load model and class indices
model = load_model(model_path)

with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Function to load image from URL
def load_image_from_url(url, target_size=(150, 150)):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    img = Image.open(BytesIO(response.content))
    img = img.resize(target_size)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]

    return img_array

# Endpoint to predict dog breed
@app.post("/predict-breed")
async def predict_breed(image_url: ImageUrl):
    try:
        img_array = load_image_from_url(image_url.url)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_breed = class_labels[predicted_class_index]
        return {"predicted_breed": predicted_breed}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app with: `uvicorn script_name:app --reload`

