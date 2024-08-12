# data_augmentation_and_class_mapping.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

def create_train_generator(train_dir, target_size=(32, 32), batch_size=256):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Use this to split training data into training and validation sets
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='training'  # Use this to specify the training subset
    )
    
    return train_generator

def create_validation_generator(train_dir, target_size=(32, 32), batch_size=256):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Use this to split training data into training and validation sets
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='validation'  # Use this to specify the validation subset
    )
    
    return validation_generator

def create_test_generator(test_dir, target_size=(32, 32), batch_size=256):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
    )
    
    return test_generator

def print_class_examples(data_generator):
    # Get the mapping from class indices to class labels
    class_labels = data_generator.class_indices
    
    # Print one image from each class
    for class_label, class_index in class_labels.items():
        # Get the indices of all the images in the class
        indices = np.where(data_generator.classes == class_index)[0]
        if len(indices) > 0:
            # Get the filename of the first image in the class
            filename = data_generator.filenames[indices[0]]
            print(f"Filename: {filename}, Class Label: {class_label}, Class Number: {class_index}")
        else:
            print(f"No images found for class: {class_label}")

if __name__ == "__main__":
    # Directories for train and test datasets
    train_dir = 'dataset_transport/dataset_transport/train'
    test_dir = 'dataset_transport/dataset_transport/test'

    train_generator = create_train_generator(train_dir)
    validation_generator = create_validation_generator(train_dir)
    test_generator = create_test_generator(test_dir)

    print("Train generator:", train_generator)
    print("Validation generator:", validation_generator)
    print("Test generator:", test_generator)

    # Print class examples from the train generator
    print("\nClass examples from the training set:")
    print_class_examples(train_generator)
