# Dog Breed Image Classification Project

## Project Overview

This project aims to classify dog breeds using a predefined dataset with images of various dog breeds. The project involves data preparation, model training using established architectures, and deploying a prediction API using FastAPI. The application is further containerized using Docker for ease of deployment.

## How to test code

### Environment Setup

To run this project, you'll need to set up your environment as follows:

1. **Ensure Python 3.11.x is Installed**

    - Verify Python version
        ```bash
      python --version
        ```
      *Note: if running on mac, python 3.12 does not work please use python 3.11 to test this*

2. **Clone the GitHub Repository**
   - Clone the repository to your local machine:
     ```bash
     git clone git@github.com:ajithpunnakula/mlzoomcap-capstone2.git
     ```

3. **Install Python Dependencies**
   
   Use pip to install necessary packages:
    - Mac/Linux: 
        ```bash
      pip install -r requirements.txt
        ```
      
    - Windows:

      ```bash
        pip install -r requirements_windows.txt
      ```    

### Build Docker Image

Run this command to build a Docker image:

```bash
docker build -t capstone2/aj-fastapi-app .
docker run -d -p 8000:8000 capstone2/aj-fastapi-app:latest
```

### Test the Deployed Docker Application

Run the following command to test the deployed Docker application:

```bash
curl -X POST "http://127.0.0.1:8000/predict-breed" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg"}'


curl -X POST "http://127.0.0.1:8000/predict-breed" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://images.unsplash.com/photo-1507146426996-ef05306b995a"}'
```
For Windows, use the following in case of trouble with curl:
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict-breed" -Method Post -Body '{"url": "https://images.unsplash.com/photo-1507146426996-ef05306b995a"}' -ContentType "application/json"


Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict-breed" -Method Post -Body '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg"}' -ContentType "application/json"
```

          
## Dataset

- **Source Dataset:** Stanford Dogs Dataset
- **Description:** The dataset includes a variety of dog breeds represented with numerous images for each breed. The dataset is prepared by splitting the data into training and validation sets in a reproducible manner using Python scripts.


## Usage

### Training

- Execute the training pipeline to prepare data, select models, and evaluate:
  ```bash
  python train_model.py
  ```

### Running the FastAPI Application Locally

- Start the FastAPI server locally to test predictions:
  ```bash
  python app.py
  ```

### Using Docker

- **Build Docker Image**
  - Create a Docker image for the application:
    ```bash
    docker build -t dog-breed-classification-app .
    ```

- **Run Docker Container**
  - Start the Docker container to serve predictions:
    ```bash
    docker run -d -p 8000:8000 dog-breed-classification-app
    ```

## Project Structure

### Key Files and Directories

- **Dockerfile**: Configuration to create the Docker image for the application.
- **app.py**: FastAPI application implementing the prediction service.
- **requirements.txt**: Lists package dependencies required for the environment.
- **data_preparation.ipynb**: Jupyter Notebook for organizing and splitting the dataset into training and validation sets.
- **class_indices.json**: Stores the mapping of class indices to dog breed names.
- **MobileNetV2_best_model.h5**: The trained model file used for predictions.
- **train_model.py**: Script for model training and evaluation with different architectures.

## Data Processing Summary

The data is organized into training and validation sets. Images are randomly shuffled and split for training and validation, with transformations applied using TensorFlow's `ImageDataGenerator` for better generalization.

### Steps Involved

1. **Data Splitting**: Data is split into training and validation sets with a ratio of 80:20.
2. **Data Augmentation**: Training data is augmented for diversity using operations like rotation, shift, shear, zoom, and flips.
3. **Normalization**: Both training and validation datasets are normalized to improve consistency during training.

## Model Training Steps

Models are trained using transfer learning on pre-trained architectures (VGG16, ResNet50, MobileNetV2).

### Steps Involved

1. **Model Creation**: Different architectures are adapted to the dog breed dataset.
2. **Model Compilation**: Models are compiled with Adam optimizer and categorical crossentropy loss.
3. **Evaluation and Selection**: Models are evaluated based on validation accuracy, selecting the best-performing model.

## API Endpoint for Dog Breed Prediction

The API leverages FastAPI to predict dog breeds from image URLs.

### Overview

- **Endpoint Definition**: A POST method to accept image URLs and return the predicted breed.
- **Model Loading**: The API loads a pre-trained MobileNetV2 model.

### Key Functions

- **Load Image From URL**: Converts image URLs into pre-processed NumPy arrays for model predictions.
- **Predict Breed**: Returns predicted dog breed based on model inference.

