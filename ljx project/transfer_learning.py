# transfer_learning.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Import the functions from data_augmentation.py
from data_augmentation import create_train_generator, create_validation_generator

# Directories for train and test datasets
train_dir = 'dataset_transport/dataset_transport/train'

train_generator = create_train_generator(train_dir)
validation_generator = create_validation_generator(train_dir)

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Create new model on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
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
model.save('transfer_learning_model.h5')

# Print model summary
model.summary()
