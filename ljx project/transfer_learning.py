import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D

# Import the functions from data_augmentation.py
from data_augmentation import create_train_generator, create_validation_generator, print_class_examples

# Directories for train and test datasets
train_dir = 'dataset_transport/dataset_transport/train'

train_generator = create_train_generator(train_dir)
validation_generator = create_validation_generator(train_dir)

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Create new model on top
x=base_model.output
x=GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x=Dense(256,activation='relu')(x)
preds=Dense(4,activation='softmax')(x)

model=Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Save the trained model
model.save('/data/saved_models/transfer_learning_model.h5')

# Print model summary
model.summary()

# Print class labels
print_class_examples(train_generator)
