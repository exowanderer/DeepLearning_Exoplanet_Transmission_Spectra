import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist

original_dim = 784
intermediate_dim = 256
latent_dim = 2
batch_size = 100
epochs = 50
epsilon_std = 1.0


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence 
            to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, original_dim) / 255.
x_test = x_test.reshape(-1, original_dim) / 255.

vae.fit(x_train,
        x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

encoder = Model(x, z_mu)

# display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(x_test, batch_size=batch_size)

def plot_scatter_solutions(y_test, z_test, alpha=0.4, s=3**2, cmap='viridis'):
    plt.figure(figsize=(6, 6))

    plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test, 
                    alpha=alpha, s=s, cmap=cmap)
    
    plt.colorbar()
    plt.show()

def plot_decoder_manifold(decoder, n_digits=15, digit_size=28, 
                            ax_min=0.05, ax_max=0.95, cmap='gray', 
                            figsize=(10, 10)):

    # n_digits: figure with n_digits x n_digits digits
    # display a 2D manifold of the digits

    # linearly spaced coordinates on the unit square were transformed
    # through the inverse CDF (ppf) of the Gaussian to produce values
    # of the latent variables z, since the prior of the latent space
    # is Gaussian
    u_grid = np.dstack(np.meshgrid(np.linspace(ax_min, ax_max, n_digits),
                                   np.linspace(ax_min, ax_max, n_digits)))

    z_grid = norm.ppf(u_grid)
    
    x_decoded = decoder.predict(z_grid.reshape(n_digits*n_digits, 2))
    x_decoded = x_decoded.reshape(n_digits, n_digits, digit_size, digit_size)

    plt.figure(figsize=figsize)
    plt.imshow(np.block(list(map(list, x_decoded))), cmap=cmap)
    plt.show()

def plot_decoder_manifold_robust(decoder, n_examples = 15, input_size = 28, 
                            ax_min = 0.05, ax_max = 0.95, cmap = 'gray', 
                            data2D = False, figsize = (10, 10)):

    # n_examples: figure with n_examples x n_examples digits
    # display a 2D manifold of the digits

    # linearly spaced coordinates on the unit square were transformed
    # through the inverse CDF (ppf) of the Gaussian to produce values
    # of the latent variables z, since the prior of the latent space
    # is Gaussian
    u_grid = np.dstack(np.meshgrid(np.linspace(ax_min, ax_max, n_examples),
                                   np.linspace(ax_min, ax_max, n_examples)))

    z_grid = norm.ppf(u_grid)
    
    # print(u_grid.shape, z_grid.shape)
    
    if data2D:
        # input_size = np.sqrt(input_size).astype(int)
        x_decoded = decoder.predict(z_grid.reshape(n_examples*n_examples, 2))
        x_decoded = x_decoded.reshape(n_examples, n_examples, 
                                        input_size, input_size)
    else:
        x_decoded = decoder.predict(z_grid.reshape(n_examples, 2))
    
    plt.figure(figsize=figsize)
    plt.imshow(np.block(list(map(list, x_decoded))), cmap=cmap)
    plt.show()

plot_scatter = True
plot_manfiold = True

if plot_scatter: plot_scatter_solutions(y_test, z_test)
if plot_manfiold: plot_decoder_manifold(decoder)