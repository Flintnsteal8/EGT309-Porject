# data_preprocessing.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, validation_dir, img_height, img_width, batch_size):
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator
