import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# Import the test generator function from data_augmentation.py
from data_augmentation import create_test_generator


# Load models
cnn_model = tf.keras.models.load_model('saved_models/cnn_model.h5')
transfer_model = tf.keras.models.load_model('saved_models/transfer_learning_model.h5')

# Load the test data generator
test_dir = 'dataset_transport/dataset_transport/test'
test_generator = create_test_generator(test_dir)

# Define class labels
class_labels = ['airplane', 'automobile', 'ship', 'truck']

# Define image paths
cnn_test_image_path = 'dataset_transport/dataset_transport/test/airplane/0499.png'
transfer_test_image_path = 'dataset_transport/dataset_transport/test/airplane/0499.png'  # Use a different image if needed

# Load and preprocess images for CNN model
cnn_img = Image.open(cnn_test_image_path).resize((32, 32))
cnn_img_array = np.array(cnn_img) / 255.0
cnn_img_array = cnn_img_array[np.newaxis, ...]  # Add batch dimension

# Make prediction using CNN model
cnn_result = cnn_model.predict(cnn_img_array)
cnn_predicted_class = np.argmax(cnn_result[0], axis=-1)

# Load and preprocess images for Transfer Learning model
transfer_img = Image.open(transfer_test_image_path).resize((32, 32))
transfer_img_array = np.array(transfer_img) / 255.0
transfer_img_array = transfer_img_array[np.newaxis, ...]  # Add batch dimension

# Make prediction using Transfer Learning model
transfer_result = transfer_model.predict(transfer_img_array)
transfer_predicted_class = np.argmax(transfer_result[0], axis=-1)


# # Predict using CNN model for classification report
cnn_predictions = cnn_model.predict(test_generator)
cnn_pred_labels = cnn_predictions.argmax(axis=1)

# Predict using transfer learning model for classification report
transfer_predictions = transfer_model.predict(test_generator)
transfer_pred_labels = transfer_predictions.argmax(axis=1)
# Plotting
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.imshow(cnn_img)
plt.title(f"CNN Model Prediction: {class_labels[cnn_predicted_class]}")

plt.subplot(1, 2, 2)
plt.imshow(transfer_img)
plt.title(f"Transfer Learning Model Prediction: {class_labels[transfer_predicted_class]}")

plt.show()

# Get true labels from the test generator
true_labels = test_generator.classes


# Classification reports
cnn_cr = classification_report(true_labels, cnn_pred_labels, target_names=test_generator.class_indices.keys())
transfer_cr = classification_report(true_labels, transfer_pred_labels, target_names=test_generator.class_indices.keys())

print("CNN Model Classification Report:\n", cnn_cr)
print("Transfer Learning Model Classification Report:\n", transfer_cr)