# data_augmentation.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for train and test datasets
train_dir = 'dataset_transport/dataset_transport/train'
test_dir = 'dataset_transport/dataset_transport/test'

# ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use this to split training data into training and validation sets
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=256,
    shuffle=True,
    class_mode='categorical',
    subset='training'  # Use this to specify the training subset
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,  # Using the same directory with subset validation for validation data
    target_size=(128, 128),
    batch_size=256,
    shuffle=True,
    class_mode='categorical',
    subset='validation'  # Use this to specify the validation subset
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=256,
    shuffle=False,
    class_mode='categorical'
)

# Example function to demonstrate the use of generators (not necessary to include in preprocessing script)
def show_batch(generator):
    import matplotlib.pyplot as plt
    images, labels = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i].argmax())
        plt.axis('off')
    plt.show()

# Uncomment to visualize a batch of training data
# show_batch(train_generator)

if __name__ == "__main__":
    # Code to save the generators if needed
    # Note: Keras generators are typically used directly in model training and not saved to files.
    
    print("Data augmentation and generators are set up.")
    print("Train generator:", train_generator)
    print("Validation generator:", validation_generator)
    print("Test generator:", test_generator)
