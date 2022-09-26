# DCGAN
An implementation of deep convolutional generative adversial networks in order to generate images of clothes based on the FASHION_MNIST dataset.
GANs use two different models, one so-called generator and one discriminator. The generator, in this case, generated images while the discriminator attempts to determine real images from fake ones.
They use different loss functions, of course, with the generators loss function depending on the discriminators predictions.
The generator gets better at generating realistic images while the discriminator gets better at predicting which ones are real.
The generator generates these images based on random 'noise'.

The model creates relatively convincing clothes fairly quickly, as seen below. The dataset itself is sometimes quite hard to classify, even as a human, after all.

![dcgan](https://user-images.githubusercontent.com/62298758/192335097-29f24e9c-bd0b-4832-9e73-a6c23179835a.gif)

The last generated image

![image_at_epoch_30](https://user-images.githubusercontent.com/62298758/192335126-267b91e7-07fc-4207-a237-7150ded2d121.png)
