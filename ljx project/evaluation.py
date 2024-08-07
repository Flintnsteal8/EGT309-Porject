# evaluation.py
import matplotlib.pyplot as plt

def plot_training_history(history_cnn, history_vgg16):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_cnn.history['accuracy'], label='CNN Training Accuracy')
    plt.plot(history_cnn.history['val_accuracy'], label='CNN Validation Accuracy')
    plt.plot(history_vgg16.history['accuracy'], label='VGG16 Training Accuracy')
    plt.plot(history_vgg16.history['val_accuracy'], label='VGG16 Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_cnn.history['loss'], label='CNN Training Loss')
    plt.plot(history_cnn.history['val_loss'], label='CNN Validation Loss')
    plt.plot(history_vgg16.history['loss'], label='VGG16 Training Loss')
    plt.plot(history_vgg16.history['val_loss'], label='VGG16 Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
