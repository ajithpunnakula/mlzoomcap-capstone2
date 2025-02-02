# Dog Breed Image Classification Project

## Project Overview

This project aims to classify dog breeds using a predefined dataset with images of various dog breeds. The project involves data preparation, model training using established architectures, and deploying a prediction API using FastAPI. The application is further containerized using Docker for ease of deployment.

## Dataset

- **Source Dataset:** Stanford Dogs Dataset
- **Description:** The dataset includes a variety of dog breeds represented with numerous images for each breed. The dataset is prepared by splitting the data into training and validation sets in a reproducible manner using Python scripts.

## Environment Setup

To run this project, you'll need to set up your environment as follows:

1. **Ensure Python 3.11+ is Installed**
   - Verify Python version:
     ```bash
     python --version
     ```

2. **Clone the GitHub Repository**
   - Clone the repository to your local machine:
     ```bash
     git clone https://github.com/yourusername/dog-breed-classification.git
     ```

3. **Install Python Dependencies**
   - Use pip to install necessary packages:
     ```bash
     pip install -r requirements.txt
     ```

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

### Example Request

To predict a dog breed, send a POST request to `/predict-breed` with the image URL included in the request body.

This setup ensures efficient and scalable deployment of the dog breed classification service, enabling accurate real-time predictions.

    ```bash
    docker build -t capstone2/aj-fastapi-app .
    docker run -d -p 8000:8000 capstone2/aj-fastapi-app:latest
    curl -X POST "http://127.0.0.1:8000/predict-breed" \
        -H "Content-Type: application/json" \
        -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg"}'
    curl -X POST "http://127.0.0.1:8000/predict-breed" \
        -H "Content-Type: application/json" \
        -d '{"url": "https://images.unsplash.com/photo-1507146426996-ef05306b995a""}'