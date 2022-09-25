import tensorflow as tf
import numpy as np


def load_data():
    (train_images, train_labels), (_,_) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")    # Originally train_images.shape[0], 28, 28
    train_images = (train_images - 127.5) / 127.5

    BUFFER_SIZE = 60000
    BATCH_SIZE = 32

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset