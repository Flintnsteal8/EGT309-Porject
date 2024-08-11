# data_augmentation.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_train_generator(train_dir, target_size=(128, 128), batch_size=256):
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

def create_validation_generator(train_dir, target_size=(128, 128), batch_size=256):
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

def create_test_generator(test_dir, target_size=(128, 128), batch_size=256):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
    )
    
    return test_generator

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
