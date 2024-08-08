# comparison.py
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load models
cnn_model = tf.keras.models.load_model('cnn_model.h5')
transfer_model = tf.keras.models.load_model('transfer_learning_model.h5')

# Load preprocessed data
X_test = pd.read_csv('X_test.csv').values.reshape(-1, 128, 128, 3)
y_test = pd.read_csv('y_test.csv').values

# Predict using CNN model
cnn_predictions = cnn_model.predict(X_test)
cnn_pred_labels = cnn_predictions.argmax(axis=1)

# Predict using transfer learning model
transfer_predictions = transfer_model.predict(X_test)
transfer_pred_labels = transfer_predictions.argmax(axis=1)

# Confusion matrices
cnn_cm = confusion_matrix(y_test, cnn_pred_labels)
transfer_cm = confusion_matrix(y_test, transfer_pred_labels)

# Classification reports
cnn_cr = classification_report(y_test, cnn_pred_labels)
transfer_cr = classification_report(y_test, transfer_pred_labels)

print("CNN Model Classification Report:\n", cnn_cr)
print("Transfer Learning Model Classification Report:\n", transfer_cr)

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cnn_cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title('CNN Model Confusion Matrix')

axes[1].imshow(transfer_cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title('Transfer Learning Model Confusion Matrix')

plt.show()
