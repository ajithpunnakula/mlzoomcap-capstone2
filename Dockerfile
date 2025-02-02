# Use the official Python image from the Docker Hub.
FROM python:3.11.10-slim

# Set the working directory inside the Docker container.
WORKDIR /app

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libc6-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file to pin dependencies.
COPY requirements_docker.txt .

# Install the required Python packages.
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the Python script and other necessary files into the Docker image.
COPY app.py .
COPY MobileNetV2_best_model.h5 .
COPY class_indices.json .

# Expose the port FastAPI will run on.
EXPOSE 8000

# Command to run the application using uvicorn server.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
