from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers

# https://www.datacamp.com/community/tutorials/generative-adversarial-networks


# x_train, y_train, x_test, y_test = load_minst_data()

# Mnist file download problem. See https://github.com/tensorflow/tensorflow/issues/10779
# Need to fix Mac certificate

(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')
print(x_train.shape)

# ---------------------------------------------------------------

optimizer = Adam(lr=0.0002, beta_1=0.5)
random_input_dim = 64


# ---------------------------------------------------------------
# Create Generator network

# TODO:
# Q? Are we initialize new instance each time when we call my_generator and my_discriminator? Not a good idea!

def my_generator():

    generator = Sequential()

    generator.add(Dense(256, input_dim=random_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    # last layer output a 28x28 image
    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return generator


def my_discriminator():

    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def my_gan(generator, discriminator):

    discriminator.trainable = False

    gan_input = Input(shape=(random_input_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gan


if __name__ == '__main__':
    print ('Hello World')
    #train(400, 128)




