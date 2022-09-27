import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

BATCH_SIZE = 32

def load_data(is_main=False):
    (train_images, train_labels), (_,_) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_images.shape)
    if not is_main:
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") # Originally [60000, 28, 28]
        train_images = (train_images - 127.5) / 127.5

        BUFFER_SIZE = 60000

        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return train_dataset

if __name__ == "__main__":
    load_data(is_main=True)