mport tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
from sklearn.metrics import classification_report
from data_augmentation import create_test_generator


# Load models
cnn_model = tf.keras.models.load_model('cnn_model.h5')
transfer_model = tf.keras.models.load_model('transfer_learning_model.h5')

# Load the test data generator
test_dir = 'dataset_transport/dataset_transport/test'
test_generator = create_test_generator(test_dir)

# Define class labels
class_labels = ['airplane', 'automobile', 'ship', 'truck']

# Define image paths
image_paths = [
    'dataset_transport/dataset_transport/test/automobile/0499.png',
    'dataset_transport/dataset_transport/test/airplane/0409.png',
    'dataset_transport/dataset_transport/test/truck/0259.png',
    'dataset_transport/dataset_transport/test/ship/0100.png'
]

# Initialize lists to hold results for plotting
cnn_predictions = []
transfer_predictions = []
images = []

# Process each image
for image_path in image_paths:
    # Load and preprocess image for CNN model
    img = Image.open(image_path).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]  # Add batch dimension

    # Make prediction using CNN model
    cnn_result = cnn_model.predict(img_array)
    cnn_predicted_class = np.argmax(cnn_result[0], axis=-1)
    cnn_predictions.append(cnn_predicted_class)

    # Make prediction using Transfer Learning model
    transfer_result = transfer_model.predict(img_array)
    transfer_predicted_class = np.argmax(transfer_result[0], axis=-1)
    transfer_predictions.append(transfer_predicted_class)

    # Save the image for plotting
    images.append(img)

# # Predict using CNN model for classification report
cnn_predictions = cnn_model.predict(test_generator)
cnn_pred_labels = cnn_predictions.argmax(axis=1)

# Predict using transfer learning model for classification report
transfer_predictions = transfer_model.predict(test_generator)
transfer_pred_labels = transfer_predictions.argmax(axis=1)

# Plotting the images with predictions
plt.figure(figsize=(20, 8))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i])
    plt.title(f"CNN Prediction: {class_labels[cnn_predictions[i]]}")
    plt.axis('off')

    plt.subplot(2, 4, i + 5)
    plt.imshow(images[i])
    plt.title(f"Transfer Prediction: {class_labels[transfer_predictions[i]]}")
    plt.axis('off')

plt.show()

# Get true labels from the test generator
true_labels = test_generator.classes

# Classification reports
cnn_cr = classification_report(true_labels, cnn_pred_labels, target_names=test_generator.class_indices.keys())
transfer_cr = classification_report(true_labels, transfer_pred_labels, target_names=test_generator.class_indices.keys())

print("CNN Model Classification Report:\n", cnn_cr)
print("Transfer Learning Model Classification Report:\n", transfer_cr)