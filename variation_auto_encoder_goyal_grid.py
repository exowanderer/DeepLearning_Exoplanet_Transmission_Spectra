import numpy as np
import os
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
# from pylab import *#;ion()
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from statsmodels.robust import scale
from time import time
from scipy.stats import norm

verbose = True
plot_scatter = False
plot_manfiold = False

if verbose: print("[INFO] Finished loading basic libraries.")

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard, EarlyStopping
from keras.losses import mean_squared_error
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist

if verbose: print("[INFO] Finished loading keras libraries.")

''' End Function Declarations '''

class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
            to the final model loss.

        This is used to determine how far "off" the latent layer
            activation is from the expected 
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = -0.5*K.sum(1+log_var-K.square(mu)-K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def nll(y_true, y_pred, y_err=None, error_type='sum_squared_error'):
    """ Negative log likelihood (Bernoulli).

        Parameters
        ----------
        y_true (ndarray): input data array
        y_pred (ndarray): current prediction array
        
        y_err (ndarray; optional): data error array. 
                                for "chisq"('sum_sq_error') `error_type`
        error_type (str; optional): select type of error to return

        Returns
        -------
        error (float): keras based error term, either summation or mean
        
        Notes
        -----
        I use a `if-elif-else` selector because I need this nll for 
            multiple keras based projects.

        [Future Development]: add error terms to nll calculation
    """

    # keras.losses.binary_crossentropy gives the mean
    #   over the last axis. we require the sum
    y_err = y_err or 1.0

    if error_type is 'sum_binary_crossentropy':
        # For latent_dim = 2 with MNIST; i.e. the keras example
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    elif error_type is 'sum_categorical_crossentropy':
        # For latent_dim > 2 with MNIST; i.e. my keras example
        return K.sum(K.categorical_crossentropy(y_true, y_pred), axis=-1)

    elif error_type is 'mean_squared_error':
        # Default expectation for regression autoencoder
        return mean_squared_error(y_true, y_pred)

    elif error_type is 'sum_squared_error':
        # Keras AutoEncoder Example specified that VAEs want to sum error
        return K.sum(K.square((y_pred - y_true)/y_err), axis=-1)

    elif error_type is 'manual_mean_squared_error':
        # Try this later in case VAE example suggestion is not the best
        return K.mean(K.square(y_pred - y_true), axis=-1)
    else:
        raise ValueError("`error_type` must be one of the following:"
                "\n['sum_binary_crossentropy', 'sum_categorical_crossentropy',"
                "\n  'mean_squared_error', 'sum_squared_error',"
                "   'manual_mean_squared_error']")

def instantiate_autoencoder(original_dim, intermediate_dim, latent_dim, 
                                epsilon_std = 1.0,
                                kernel_initializer = 'he_normal',
                                latent_activation = 'relu', 
                                output_activation = 'sigmoid',
                                return_all = False):
    ''' Keras AutoEncoder

        Parameters
        ----------
        original_dim (int): size of the input data per sample
        intermediate_dim (int): size of the hidden layer 
                                between input and latent layers
        latent_dim (int): size of latent layer: number of `neurons` with which
                        describe the data over (similar to "num_pca_vectors")
        latent_activation (str; optional): hidden layer activation function
        output_activation (str; optional): Encoder output activation function

    '''
    # Input data layer
    inputs = Input(shape = (original_dim,))

    # Hidden layer for variational noise term
    var_hidden = Dense(intermediate_dim, 
                        kernel_initializer=kernel_initializer,
                        activation=latent_activation)(inputs)

    # Mean and Variance of the `noise` layer for the `variational` behaviour
    #   (?) Mean layer acts as encoder here
    z_mu = Dense(latent_dim, kernel_initializer=kernel_initializer)(var_hidden)
    z_log_var = Dense(latent_dim, 
                kernel_initializer=kernel_initializer)(var_hidden)

    # Add KL-Divergences loss over the variational layer
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

    # Gaussian noise term to add variational behaviour
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    # Scaling layer for random noise variations
    eps = Input(tensor = K.random_normal(stddev=epsilon_std,
                shape = (K.shape(inputs)[0], latent_dim)))

    # Multiplicative combination of scaling layer and Gaussian noise layer
    z_eps = Multiply()([z_sigma, eps])

    # Additive combination of encoder output and variational layer
    encoder = Add()([z_mu, z_eps])

    # Decoder layer: from latent dimensions to generative predictions
    decoder = Sequential([Dense(intermediate_dim, input_dim = latent_dim, 
                                kernel_initializer=kernel_initializer,
                                activation = latent_activation),
                          Dense(original_dim, 
                                kernel_initializer=kernel_initializer,
                                activation = output_activation)
                         ])
    
    # Output: encode, add variations, decode
    outputs = decoder(encoder)

    vae_model = Model(inputs = [inputs, eps], outputs = outputs)

    if return_all:
        return vae_model, inputs, outputs, eps, decoder, z_mu
    else:
        return vae_model

def plot_samples(labels, wavelengths = None, n_samples = 10):

    n_spectra, n_waves = labels.shape

    if wavelengths is None:
        wavelengths = np.arange(n_waves)

    idx_ = np.random.choice(np.arange(n_spectra), 
                                size = n_samples, 
                                replace = False)

    plt.plot(wavelengths, labels[idx_].T)

''' End Function Declarations '''
wavelengths = None
waves_use = None

if verbose: print("[INFO] Load data from harddrive.")

spectra_filenames = glob('transmission_spectral_files/trans*')
spectral_grid = {}

for fname in tqdm(filenames):
    key = '_'.join(fname.split('/')[-1].split('_')[1:7])
    info_now = np.loadtxt(fname)
    if wavelengths is None: wavelengths = info_now[:,0]
    if waves_use is None: waves_use = wavelengths < 5.0
    spectral_grid[key] = info_now[:,1][waves_use]

if verbose: print("[INFO] Assigning input values onto `labels` and `features`")

n_waves = waves_use.sum()
labels = np.zeros((len(spectra_filenames), n_waves))
features = np.zeros((len(spectra_filenames), len(key.split('_'))))

for k, (key,val) in enumerate(spectral_grid.items()): 
    labels[k] = val
    features[k] = np.array(key.split('_')).astype(float)

if verbose: print("[INFO] Computing train test split over indices "
                    "with shuffling")

test_size = 0.2
idx_train, idx_test = train_test_split(np.arange(len(spectra_filenames)), 
                                        test_size=test_size)

if verbose: print("[INFO] Assigning x_train, y_train, x_test, y_test "
                    "from `idx_train` and `idx_test`")

''' Organize input data for autoencoder '''
x_train = features[idx_train]
y_train = labels[idx_train]

x_test = features[idx_test]
y_test = labels[idx_test]

if verbose: print('Computing Median Spectrum')
y_train_med = np.median(y_train, axis=0)

if verbose: print('Computing Median Average Deviation Spectrum')
y_train_mad = scale.mad(y_train, axis=0)
min_train_mad = 1e-6

''' Initiate and Fit AutoEncoder '''
original_dim = waves_use.sum()
intermediate_dim = 256
latent_dim = 6
batch_size = 100
epochs = 50
epsilon_std = 1.0
shuffle = True
latent_activation = 'relu'
output_activation = 'sigmoid'
return_all = True

vae_model, inputs, outputs, eps, decoder, z_mu = instantiate_autoencoder(
                                       original_dim = original_dim,
                                       intermediate_dim = intermediate_dim,
                                       latent_dim = latent_dim, 
                                       latent_activation = latent_activation,
                                       output_activation = output_activation,
                                       epsilon_std = epsilon_std,
                                       return_all = return_all)

optimizer = 'adam'
vae_model.compile(optimizer=optimizer, loss=nll)
#, metrics=['val_loss', 'val_acc', 'acc']

# y_train_fit = ((y_train - y_train_med) / (y_train_mad + min_train_mad))
# y_test_fit = ((y_test - y_train_med) / y_train_mad + min_train_mad)

y_train_fit = (y_train - y_train_min) / (y_train_max - y_train_min)
y_test_fit = (y_test - y_train_min) / (y_train_max - y_train_min)

# Swapped x_train, x_test <--> y_train, y_test 
#   because this is an inversion problem
callback_list = []

time_stamp = int(time())

use_model_checkpoint = True
use_tensorboard = True
use_lr_reduce = True
use_early_stopping = True

if use_model_checkpoint:
    checkpoint_directory = 'checkpoints/'
    optimization_mode = 
    monitor = 
    verbose = 1
    model_checkpoint_monitor = 'loss'
    abs_weight_path = os.path.dirname(os.path.abspath(checkpoint_directory))
    if not os.path.exists(abs_weight_path):
        print("Creating {}".format(abs_weight_path))
        os.mkdir(abs_weight_path)

    model_checkpoint_opt_mode = 'auto'
    model_checkpoint = ModelCheckpoint(checkpoint_directory, 
                                        verbose=verbose, 
                                        mode=model_checkpoint_opt_mode, 
                                        monitor=model_checkpoint_monitor, 
                                        save_best_only=True, 
                                        save_weights_only=True)
    
    callback_list.append(model_checkpoint)

if use_lr_reduce:
    is_timeseries = True
    if is_timeseries:
        lr_reduce_factor = 1. / np.cbrt(2)
    else:
        lr_reduce_factor = 1. / np.sqrt(2)

    reduce_lr_monitor = 'loss'
    reduce_lr_opt_mode='auto'
    reduce_lr = ReduceLROnPlateau(  monitor = reduce_lr_monitor, 
                                    patience = 100, 
                                    mode = reduce_lr_opt_mode,
                                    factor = lr_reduce_factor, 
                                    cooldown = 0, 
                                    min_lr = 1e-4, 
                                    verbose = 2)

    callback_list.append(reduce_lr)

if use_tensorboard:
    tensorboard_logdir = './logs/log-{}'
    
    if '{}' in tensorboard_logdir:
        tensorboard_logdir = tensorboard_logdir.format(time_stamp)
    
    tensorboard = TrainValTensorboard(log_dir=tensorboard_logdir, 
                                        write_graph=False)
    
    callback_list.append(tensorboard)

if use_early_stopping:
    early_stopping_monitor = 'val_loss'
    early_stopping = EarlyStopping(monitor = early_stopping_monitor, 
                                    min_delta = 0,
                                    patience = 10, 
                                    verbose = 1, 
                                    mode = 'auto')
    
    callback_list.append(early_stopping)

vae_history = vae_model.fit(y_train_fit, y_test_fit,
                            shuffle = shuffle,
                            epochs = epochs,
                            batch_size = batch_size,
                            validation_data = (y_test_fit, y_test_fit),
                            callback = callback_list)

data_dir = '/SaveFiles/'
save_dir = os.environ['HOME'] + data_dir

savename_tmplt = save_dir + 'grid_vae_{}_{}'

if verbose: print("[INFO] Saving fitted model every way that I know how.")

if verbose: print("[INFO] Saving full model")
vae_model.save(savename_tmplt.format(time_stamp, 'full_model_save.hdf5'))

if verbose: print("[INFO] Saving model weights")
vae_model.save_weights(savename_tmplt.format(time_stamp, 
                        'model_weights_save.hdf5'))

if verbose: print("[INFO] Saving model instance")
joblib.dump(vae_model, savename_tmplt.format(time_stamp, 
                        'full_model.joblib.save'))

if verbose: print("[INFO] Saving model history output")
joblib.dump(vae_history, savename_tmplt.format(time_stamp, 
                        'model_history.joblib.save'))

# Encoder layer for use with making predictions
encoder = Model(inputs, z_mu)

# Display a plot of the latent space; e.g. new digits, faces, (here) spectra.
z_test = encoder.predict(y_test, batch_size=batch_size)

def plot_history(history, figsize=(12,12), skipfirst=True):
    
    skipfirst = np.int(skipfirst)

    plt.figure(figsize=figsize)
    plt.plot(history.epoch[skipfirst:], history.history['loss'][skipfirst:])
    plt.plot(history.epoch[skipfirst:], history.history['val_loss'][skipfirst:])
    plt.show()

''' Plotting Functions from Predictions and Decoder '''
def plot_scatter_solutions(x_test, z_test, z_element0 = 0, z_element1 = 1,
                            alpha = 0.4, s = 3**2, cmap = 'viridis'):
    
    plt.figure(figsize=(6, 6))

    plt.scatter(z_test[:, z_element0], z_test[:, z_element1], c=x_test, 
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


if plot_scatter: 
    for z_ele1 in range(1,z_test.shape[1]): 
        for x_test_feat in x_test.T: 
            plot_scatter_solutions(x_test_feat, z_test, z_element1=z_ele1)

if plot_manfiold: 
    plot_decoder_manifold(decoder, n_examples=10, input_size=y_test.shape[1])