# cnn_model_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Import the functions from data_augmentation.py
from data_augmentation import create_train_generator, create_validation_generator, create_test_generator

# Directories for train and test datasets
train_dir = 'dataset_transport/dataset_transport/train'
test_dir = 'dataset_transport/dataset_transport/test'

train_generator = create_train_generator(train_dir)
validation_generator = create_validation_generator(train_dir)
# test_generator can be used later if needed

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Adjust output for the number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Save the trained model
model.save('cnn_model.h5')

# Print model summary
model.summary()
