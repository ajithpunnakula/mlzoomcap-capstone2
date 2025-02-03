# Dog Breed Image Classification Project

## Project Overview

This project aims to classify dog breeds using a predefined dataset with images of various dog breeds. The project involves data preparation, model training using established architectures, and deploying a prediction API using FastAPI. The application is further containerized using Docker for ease of deployment.

## Dataset

- **Source Dataset:** [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
- **Description:** The dataset includes a variety of dog breeds represented with numerous images for each breed. The dataset is prepared by splitting the data into training and validation sets in a reproducible manner using Python scripts.

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

### Using Docker

- **Build Docker Image**
  - Run this command to build a Docker image:

    ```bash
    docker build -t capstone2/aj-fastapi-app .
    ```
- **Run Docker Container**
  - Start the Docker container to serve predictions:
    ```bash
    docker run -d -p 8000:8000 capstone2/aj-fastapi-app:latest
    ```


### Test the Deployed Docker Application
There are two ways to test the model deployed in Docker application

1. Run the python file test.py
    ```bash
        python test.py
    ```
This tests the dog breed of this image - *https://images.unsplash.com/photo-1507146426996-ef05306b995a*

2. Run the following command to test the deployed Docker application:

    ```bash
    curl -X POST "http://127.0.0.1:8000/predict-breed" \
        -H "Content-Type: application/json" \
        -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg"}'


    curl -X POST "http://127.0.0.1:8000/predict-breed" \
        -H "Content-Type: application/json" \
        -d '{"url": "https://images.unsplash.com/photo-1507146426996-ef05306b995a"}'
    ```


   For Windows, use the following in case of trouble with `curl`:

   ```bash
   Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict-breed" -Method Post -Body '{"url": "https://images.unsplash.com/photo-1507146426996-ef05306b995a"}' -ContentType "application/json"

   Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict-breed" -Method Post -Body '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg"}' -ContentType "application/json"
   ```

## How to run the training code

### Download dataset and unzip
  - Download using the following curl command or
    ```bash
    #!/bin/bash
    curl -L -o ~/Downloads/stanford-dogs-dataset.zip\
      https://www.kaggle.com/api/v1/datasets/download/jessicali9530/stanford-dogs-dataset
    ```
  - Download directly from kaggle website
    You can access the Stanford Dogs Dataset on Kaggle using the following link: [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

  - unzip the dataset in the same folder as the cloned code. Note that I have added a **-copy** at the end of the extracted folder
    ```bash
    unzip stanford-dogs-dataset.zip -d ./stanford-dogs-dataset-copy
    ```


### Training

- After the dataset has been downloaded and extracted with above instructions, Execute the training pipeline to prepare data, select models, and evaluate:
  ```bash
  python train.py
  ```

### Running the FastAPI Application Locally (use docker instructions above instead of this)

- Start the FastAPI server locally to test predictions:
  ```bash
  python app.py
  ```

## Project Structure

### Key Files and Directories

- **Dockerfile**: Configuration to create the Docker image for the application.
- **app.py**: FastAPI application implementing the prediction service.
- **requirements.txt**: Lists package dependencies required for the environment.
- **train.ipynb**: Jupyter Notebook for organizing and splitting the dataset into training and validation sets.
- **class_indices.json**: Stores the mapping of class indices to dog breed names.
- **MobileNetV2_best_model.h5**: The trained model file used for predictions.
- **train.py**: Script for model training and evaluation with different architectures.
- **test.py**: Script to test the deployed docker application

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

## Model Evaluation Summary

This document summarizes the performance of three models evaluated on the dataset and identifies the best-performing model.

### Evaluated Models

Three different models were trained and evaluated:

1. **VGG16**
2. **ResNet50**
3. **MobileNetV2**

### Performance Metrics

The performance of each model was measured using validation loss and validation accuracy.

  | Model        | Validation Loss | Validation Accuracy |
  |--------------|-----------------|---------------------|
  | VGG16        | 3.1033          | 0.2556              |
  | ResNet50     | 4.7058          | 0.0209              |
  | MobileNetV2  | **2.2696**      | **0.4476**          |

### Best Model

After comparing the performance metrics, the **MobileNetV2** model was selected as the best model based on the following criteria:

- **Lowest Validation Loss**: 2.2696
- **Highest Validation Accuracy**: 0.4476

These results indicate that MobileNetV2 has a superior generalization capability on the validation dataset compared to VGG16 and ResNet50.

### Conclusion

MobileNetV2 is recommended for further use and deployment due to its effective balance between accuracy and loss. Additionally, MobileNetV2's architecture favors efficiency and speed, which is beneficial for deployment in resource-constrained environments.

### Future Work

- Further tuning of MobileNetV2's hyperparameters could lead to additional improvements in accuracy.
- Evaluation on a test set should be conducted to confirm these findings.
- Consider experimenting with ensemble methods or augmenting the dataset for potentially better results.

### References

- [MobileNetV2 Architecture](https://arxiv.org/abs/1801.04381)
- [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)



## FastAPI Dog Breed Prediction Application

This application provides an API endpoint for predicting the breed of a dog given the image URL. It leverages a pre-trained MobileNetV2 model to perform the classification.

### Overview

The application is built using FastAPI, a modern web framework for building APIs with Python. It accepts a JSON request with an image URL, processes the image, and returns the predicted dog breed.

### Key Components

- **Imports**:
  - `FastAPI`, `HTTPException` for building the API.
  - `BaseModel` from `pydantic` to define the structure of incoming request data.
  - `Image` from `PIL`, `BytesIO`, and `requests` for handling image processing.
  - `load_model` from TensorFlow Keras to load the machine learning model.
  - `numpy` for numerical operations.

- **Pydantic Model**:
  - `ImageUrl`: A class that defines the expected format of the request body, specifically that it should contain a `url` key with a string value.

- **Model and Class Indices Loading**:
  - Load the pre-trained model from a `.h5` file.
  - Load class indices from a JSON file, mapping model outputs to human-readable labels.

- **Image Processing**:
  - `load_image_from_url`: A function to download and process the image from the provided URL. The image is resized, normalized, and converted into an array format suitable for input into the model.

- **API Endpoint**:
  - `/predict-breed`: Accepts POST requests containing an image URL. It processes the image, performs prediction with the model, and returns the predicted breed. Handles exceptions by returning a 400 HTTP error code with a description of the problem.

### Usage Example

To test the endpoint, send a POST request to `/predict-breed` with the following JSON payload format:
(use the test instructions above, the below is just for quick reference)
```bash
curl -X POST "http://127.0.0.1:8000/predict-breed" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg"}'


curl -X POST "http://127.0.0.1:8000/predict-breed" \
    -H "Content-Type: application/json" \
    -d '{"url": "https://images.unsplash.com/photo-1507146426996-ef05306b995a"}'
```
