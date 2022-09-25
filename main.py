import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import imageio
import numpy as np
import PIL.Image as Image
import time

from data_processing import load_data, BATCH_SIZE
from models import make_generator_model, make_discriminator_model


def main():
    train_dataset = load_data()
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)   # Ideal prediction on real image would be a matrix of ones
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # Ideal prediction on fake image would be a matrix of zeros

        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):    # Ideal would be for the discriminator to classify predictions as true (1s)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)  # Contains BatchNormalization() layers -> training=True should be specified
            
            real_output = discriminator(images, training=True)  # Contains Dropout() layers -> training=True should be specified
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    
    def train(dataset, epochs):
        for epoch in range(epochs):
            tic = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            generate_and_save_images(
                generator,
                epoch + 1,
                seed
            )

            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print(f"Epoch: {epoch}, Time: {time.time() - tic}")

        generate_and_save_images(
                    generator,
                    epoch + 1,
                    seed
                )

    
    def generate_and_save_images(model, epoch, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4,4,i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")

        plt.savefig(f"Images/image_at_epoch_{epoch}")
        plt.show()  # Must close the window before code continues


    train(train_dataset, EPOCHS)
    

if __name__ == "__main__":
    main()