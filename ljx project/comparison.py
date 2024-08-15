from flask import Flask, render_template_string, render_template
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import io
import base64
from sklearn.metrics import classification_report
from data_augmentation import create_test_generator

app = Flask(__name__)

@app.route('/')
def home():
    # Directly call the compare_models function to render the comparison page
    return compare_models()

@app.route('/compare')
def compare_models():
    # Load models
    cnn_model = tf.keras.models.load_model('/app/saved_models/cnn_model.h5')
    transfer_model = tf.keras.models.load_model('/app/saved_models/transfer_learning_model.h5')

    # Load the test data generator
    test_dir = 'dataset_transport/dataset_transport/test'
    test_generator = create_test_generator(test_dir)

    # Define class labels and corresponding image paths
    class_labels = ['airplane', 'automobile', 'ship', 'truck']
    image_paths = {
        'airplane': 'dataset_transport/dataset_transport/test/airplane/0499.png',
        'automobile': 'dataset_transport/dataset_transport/test/automobile/0017.png',
        'ship': 'dataset_transport/dataset_transport/test/ship/0123.png',
        'truck': 'dataset_transport/dataset_transport/test/truck/0023.png'
    }

    predictions = []

    for label in class_labels:
        # Load and preprocess image for CNN model
        img = Image.open(image_paths[label]).resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = img_array[np.newaxis, ...]

        # Make predictions using both models
        cnn_result = cnn_model.predict(img_array)
        transfer_result = transfer_model.predict(img_array)

        cnn_predicted_class = np.argmax(cnn_result[0], axis=-1)
        transfer_predicted_class = np.argmax(transfer_result[0], axis=-1)

        # Save the plot to a buffer
        img_buf = io.BytesIO()
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"CNN Prediction: {class_labels[cnn_predicted_class]}")

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.title(f"Transfer Learning Prediction: {class_labels[transfer_predicted_class]}")

        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.getvalue()).decode()

        # Store the predictions and corresponding images
        predictions.append({
            'class': label,
            'img_data': img_data,
            'cnn_prediction': class_labels[cnn_predicted_class],
            'transfer_prediction': class_labels[transfer_predicted_class]
        })

    # Get true labels from the test generator
    true_labels = test_generator.classes

    # Classification reports
    cnn_predictions = cnn_model.predict(test_generator)
    cnn_pred_labels = cnn_predictions.argmax(axis=1)
    transfer_predictions = transfer_model.predict(test_generator)
    transfer_pred_labels = transfer_predictions.argmax(axis=1)

    cnn_cr = classification_report(true_labels, cnn_pred_labels, target_names=test_generator.class_indices.keys())
    transfer_cr = classification_report(true_labels, transfer_pred_labels, target_names=test_generator.class_indices.keys())

    # Specify the full path to the comparison.html file
    html_file_path = r'C:\Users\Edric\Downloads\EGT309-Project\ljx project\comparison.html'

    # Read the HTML template directly from the file system
    with open(html_file_path) as f:
        html_template = f.read()

    # Render the template with the predictions and classification report
    return render_template_string(html_template, predictions=predictions, cnn_cr=cnn_cr, transfer_cr=transfer_cr)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)