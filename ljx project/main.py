# main.py
from data_augmentation import create_generators
from model_training import create_cnn_model, create_vgg16_model, train_model
from evaluation import plot_training_history

train_dir = 'ljx project/dataset_transport/dataset_transport/train'
validation_dir = 'ljx project/dataset_transport/dataset_transport/test'
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

train_generator, validation_generator = create_generators(train_dir, validation_dir, img_height, img_width, batch_size)

cnn_model = create_cnn_model((img_height, img_width, 3))
vgg16_model = create_vgg16_model((img_height, img_width, 3))

history_cnn = train_model(cnn_model, train_generator, validation_generator, epochs)
history_vgg16 = train_model(vgg16_model, train_generator, validation_generator, epochs)

plot_training_history(history_cnn, history_vgg16)
