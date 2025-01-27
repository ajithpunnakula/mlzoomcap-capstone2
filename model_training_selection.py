# data preparation

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

# Set the path to the dataset
train_dir = '/Users/apunnakula/AJ/mlzoomcap-capstone2/stanford-dogs-dataset-copy/train'
validation_dir = '/Users/apunnakula/AJ/mlzoomcap-capstone2/stanford-dogs-dataset-copy/validation'

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Normalization for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)


# model selection and training

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

def create_model(model_name):
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    else:
        raise ValueError("Model not recognized")

    # Freeze the base model
    base_model.trainable = False

    # Create a new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    return model

# Define a function to compile, train, and evaluate models
def train_evaluate_model(model_name):
    model = create_model(model_name)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(f'{model_name}_best_model.h5', save_best_only=True),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f'{model_name} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
    return val_accuracy

# Train and evaluate each model
model_names = ['VGG16', 'ResNet50', 'MobileNetV2']
val_accuracies = []

for model_name in model_names:
    accuracy = train_evaluate_model(model_name)
    val_accuracies.append(accuracy)

# Find the best model
best_model_index = np.argmax(val_accuracies)
print(f'The best model is {model_names[best_model_index]} with an accuracy of {val_accuracies[best_model_index]}')
