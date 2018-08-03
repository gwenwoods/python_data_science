from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
import numpy as np
import matplotlib.pyplot as plt

# https://www.datacamp.com/community/tutorials/generative-adversarial-networks
# ---------------------------------------------------------------
# global variables

optimizer = Adam(lr=0.0002, beta_1=0.5)
random_input_dim = 64 # size of initial noise array as generator input


# ---------------------------------------------------------------
# Create Generator network

# TODO:
# Q? Are we initialize new instance each time when we call my_generator and my_discriminator? Not a good idea!

def my_generator():

    img_generator = Sequential()

    img_generator.add(Dense(256, input_dim=random_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    img_generator.add(LeakyReLU(0.2))

    img_generator.add(Dense(512))
    img_generator.add(LeakyReLU(0.2))

    img_generator.add(Dense(1024))
    img_generator.add(LeakyReLU(0.2))

    # last layer output a 28x28 image
    img_generator.add(Dense(784, activation='tanh'))
    img_generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return img_generator


def my_discriminator():

    img_discriminator = Sequential()

    img_discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    img_discriminator.add(LeakyReLU(0.2))
    img_discriminator.add(Dropout(0.3))

    img_discriminator.add(Dense(512))
    img_discriminator.add(LeakyReLU(0.2))
    img_discriminator.add(Dropout(0.3))

    img_discriminator.add(Dense(256))
    img_discriminator.add(LeakyReLU(0.2))
    img_discriminator.add(Dropout(0.3))

    img_discriminator.add(Dense(1, activation='sigmoid'))
    img_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return img_discriminator


def my_gan(generator, discriminator):

    discriminator.trainable = False

    gan_input = Input(shape=(random_input_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    img_gan = Model(inputs=gan_input, outputs=gan_output)
    img_gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return img_gan

def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(16, 16)):

    test_noise = np.random.normal(0, 1, size=[examples, random_input_dim])
    generated_images = generator.predict(test_noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
    # plt.show()



if __name__ == '__main__':

    print('GAN network')

    # -------------------------------------------------------------------------------------------
    # Read in data
    # x_train, y_train, x_test, y_test = load_minst_data()

    # Mnist file download problem. See https://github.com/tensorflow/tensorflow/issues/10779
    # Need to fix Mac certificate

    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')
    print(x_train.shape)

    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape(60000, 784)

    # -------------------------------------------------------------------------------------------
    # Training

    batch_size = 128
    batch_count = int(x_train.shape[0]/batch_size)

    print('Meow')
    print(len(np.random.randint(0, x_train.shape[0], size=batch_size)))

    #

    generator = my_generator()
    discriminator = my_discriminator()

    gan = my_gan(generator, discriminator)

    epochs = 200
    for e in range(1, epochs + 1):
        print('epoch ', e)

        for batch_idx in range (0, batch_count):
            # -----------------------------------------------------------------------------
            # for each epoch, we need to create inputs for both generator and discriminator
            # input for generator:
            noise = np.random.normal(0, 1, size=[batch_size, random_input_dim])
            #print (' noise shape: ', noise.shape)

            # keras model predict API: predict(x, batch_size=None, verbose=0, steps=None)

            image_batch_fake = generator.predict(noise)  # will be used as part of discriminator input
            # print (image_batch_fake.shape)

            # real input for discriminator:
            #   randomly selected batch_size elements from the training set (sample with replacement)
            # image_batch_real = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            image_batch_real = x_train[batch_idx*128: (batch_idx+1)*128]

            image_batch_all = np.concatenate([image_batch_real, image_batch_fake])

            # Lables for the combined real/fake image_batch_all
            # real as 1 (or close to 1), fake is 0
            y_all = np.zeros(2*batch_size)
            y_all[:batch_size] = 0.9

            # Train the discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(image_batch_all, y_all)

            # Train the generator
            noise = np.random.normal(0, 1, size=[batch_size, random_input_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 10 == 0:
            plot_generated_images(e, generator)






