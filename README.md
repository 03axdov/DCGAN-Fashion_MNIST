# DCGAN
An implementation of deep convolutional generative adversial networks in order to generate handwritten digits based on the MNIST dataset.
GANs use two different models, one so-called generator and one discriminator. The generator, in this case, generated images while the discriminator attempts to determine real images from fake ones.
They use different loss functions, of course, with the generators loss function depending on the discriminators predictions.
The generator gets better at generating realistic images while the discriminator gets better at predicting which ones are real.

The generator generates these images based on random 'noise' as seen below.

![from_noise](https://user-images.githubusercontent.com/62298758/192326591-f5a07970-7a4c-49d2-ba0a-b9c465a89c4e.png)

The model creates relatively convincing digits fairly quickly, as seen below.

![dcgan](https://user-images.githubusercontent.com/62298758/192326434-e743a2ac-584d-4ff0-b727-3bd6618cea2b.gif)

The last generated image

![image_at_epoch_50](https://user-images.githubusercontent.com/62298758/192327228-3f57b565-d73d-4d8b-8df5-ccc1951377ad.png)
