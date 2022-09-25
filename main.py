import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import imageio
import numpy as np
import PIL.Image as Image
import time

from data_processing import load_data
from models import make_generator_model


def main():
    train_dataset = load_data()

    generator = make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise)

    img = Image.fromarray(generated_image.numpy())
    img.show()


if __name__ == "__main__":
    main()