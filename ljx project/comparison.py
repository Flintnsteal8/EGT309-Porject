# comparison.py
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
# Import the test generator function from data_augmentation.py
from data_augmentation import create_test_generator

print(tf.__version__)
# Load models
cnn_model = tf.keras.models.load_model('cnn_model.h5')
transfer_model = tf.keras.models.load_model('transfer_learning_model.h5')

# Load the test data generator
test_dir = 'dataset_transport/dataset_transport/test'
test_generator = create_test_generator(test_dir)

# # Predict using CNN model
# cnn_predictions = cnn_model.predict(test_generator)
# cnn_pred_labels = cnn_predictions.argmax(axis=1)

# # Predict using transfer learning model
# transfer_predictions = transfer_model.predict(test_generator)
# transfer_pred_labels = transfer_predictions.argmax(axis=1)

# # Get true labels from the test generator
# true_labels = test_generator.classes

# # Confusion matrices
# cnn_cm = confusion_matrix(true_labels, cnn_pred_labels)
# transfer_cm = confusion_matrix(true_labels, transfer_pred_labels)

# # Classification reports
# cnn_cr = classification_report(true_labels, cnn_pred_labels, target_names=test_generator.class_indices.keys())
# transfer_cr = classification_report(true_labels, transfer_pred_labels, target_names=test_generator.class_indices.keys())

# print("CNN Model Classification Report:\n", cnn_cr)
# print("Transfer Learning Model Classification Report:\n", transfer_cr)

# # Plot confusion matrices
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# axes[0].imshow(cnn_cm, interpolation='nearest', cmap=plt.cm.Blues)
# axes[0].set_title('CNN Model Confusion Matrix')

# axes[1].imshow(transfer_cm, interpolation='nearest', cmap=plt.cm.Blues)
# axes[1].set_title('Transfer Learning Model Confusion Matrix')

# plt.show()
