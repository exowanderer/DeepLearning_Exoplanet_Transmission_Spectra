from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

import numpy

import numpy as np
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
UseGridSearchCV = False
UseRandomizedSearchCV = True
plot_scatter = False
plot_manfiold = False

if verbose: print("[INFO] Finished loading basic libraries.")

from keras import backend as K
from keras.losses import mean_squared_error
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist

if verbose: print("[INFO] Finished loading keras libraries.")

''' End Function Declarations '''
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

class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
            to the final model loss.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = -0.5*K.sum(1+log_var-K.square(mu)-K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

def instantiate_autoencoder(original_dim, intermediate_dim, latent_dim, 
                            n_hidden_layers = 1,
                            kernel_initializer='he_normal',
                            hidden_layer_ratio = 2,
                            epsilon_std = 1.0, 
                            latent_activation='relu', 
                            output_activation='sigmoid',
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

    # Adds n_hidden_layers - 1, with reduced size `intermediate_dim`
    for _ in range(n_hidden_layers-1):
        intermediate_dim = np.ceil(intermediate_dim // hidden_layer_ratio)
        var_hidden = Dense(int(intermediate_dim), 
                            kernel_initializer=kernel_initializer,
                            activation=latent_activation)(var_hidden)
        
        if intermediate_dim > latent_dim:
            print('Intermediate_dim should be larger then latent_dim (?)\n'
                  '\tintermediate_dim: {}\n\tlatent_dim: {}'.format(
                        intermediate_dim, latent_dim))

    # Mean and Variance of the `noise` layer for the `variational` behaviour
    #   (?) Mean layer acts as encoder here
    z_mu = Dense(latent_dim, kernel_initializer=kernel_initializer)(var_hidden)
    z_log_var = Dense(latent_dim, 
                    kernel_initializer=kernel_initializer)(var_hidden)

    # KL-Divergences over the Mean and Variance of the `noise` layer
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
    decoder = Sequential()
    decoder.add(Dense(intermediate_dim, 
                        input_dim = latent_dim, 
                        kernel_initializer=kernel_initializer,
                        activation = latent_activation))

    # Adds n_hidden_layers - 1, with increased size `intermediate_dim`
    for _ in range(n_hidden_layers-1):
        intermediate_dim = np.ceil(intermediate_dim * hidden_layer_ratio)
        decoder.add(Dense(int(intermediate_dim), 
                          kernel_initializer=kernel_initializer,
                          activation = latent_activation))

    decoder = Dense(original_dim, 
                    kernel_initializer=kernel_initializer,
                    activation = output_activation)
    
    # Output: encode, add variations, decode
    outputs = decoder(encoder)

    vae_model = Model(inputs = [inputs, eps], outputs = outputs)

    if return_all:
        return vae_model, inputs, outputs, eps, decoder, z_mu
    else:
        return vae_model

''' End Primary Function Declarations '''
wavelengths = None
waves_use = None

if verbose: print("[INFO] Load data from harddrive.")

spectra_filenames = glob('transmission_spectral_files/trans*')
spectral_grid = {}

for fname in tqdm(spectral_filenames):
    key = '_'.join(fname.split('/')[-1].split('_')[1:7])
    info_now = np.loadtxt(fname)
    if wavelengths is None: wavelengths = info_now[:,0]
    if waves_use is None: waves_use = wavelengths < 5.0
    spectral_grid[key] = info_now[:,1][waves_use]

''' Organize input data for autoencoder '''
if verbose: print("[INFO] Assigning input values onto `labels` and `features`")

n_waves = waves_use.sum()
labels = np.zeros((len(spectral_filenames), n_waves))
features = np.zeros((len(spectral_filenames), len(key.split('_'))))

for k, (key,val) in enumerate(spectral_grid.items()): 
    labels[k] = val
    features[k] = np.array(key.split('_')).astype(float)

if verbose: print("[INFO] Computing train test split over indices "
                    "with shuffling")

test_size = 0.2
idx_train, idx_test = train_test_split(np.arange(len(spectral_filenames)), 
                                        test_size=test_size)

if verbose: print("[INFO] Assigning x_train, y_train, x_test, y_test "
                    "from `idx_train` and `idx_test`")

x_train = features[idx_train]
y_train = labels[idx_train]

x_test = features[idx_test]
y_test = labels[idx_test]

if verbose: print('Computing Median Spectrum')
y_med = np.median(y_train, axis=0)

if verbose: print('Computing Median Average Deviation Spectrum')
y_mad = scale.mad(y_train, axis=0)

''' Initiate and Fit AutoEncoder via Grid Search'''
# y_train_fit = ((y_train - y_train_med) / (y_train_mad + min_train_mad))
# y_test_fit = ((y_test - y_train_med) / y_train_mad + min_train_mad)

y_train_fit = (y_train - y_train_min) / (y_train_max - y_train_min)
y_test_fit = (y_test - y_train_min) / (y_train_max - y_train_min)

# Create model
batch_size = 100
epochs = 50
# verbose = True

def create_model(intermediate_dim, latent_dim, n_hidden_layers, 
                 original_dim=waves_use.sum(), latent_activation='relu', 
                 output_activation='sigmoid', kernel_initializer='he_normal', 
                 optimizer = 'adam'):
    ''' Variational AutoEncoder Wrapper Function for GridSearchCV Usage '''

    vae_model = instantiate_autoencoder(original_dim = original_dim,
                                        intermediate_dim = intermediate_dim,
                                        latent_dim = latent_dim, 
                                        n_hidden_layers = n_hidden_layers,
                                        latent_activation = latent_activation,
                                        output_activation = output_activation,
                                        return_all = False)

    vae_model.compile(optimizer=optimizer, loss=nll, metrics=['acc'])
    #, 'val_loss', 'val_acc'

    return vae_model

model = KerasClassifier(build_fn = create_model, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        # validation_data = (y_test_fit, y_test_fit),
                        verbose = verbose)

if UseGridSearchCV:
    search_type = ''
    # 
    n_cv = 5
    n_hidden_layers = np.arange(1,6) # 5 options
    intermediate_dims = [128, 256, 512] # 3 options
    latent_dims = np.arange(2,8) # 9 options
    latent_activations = ['relu', 'elu'] # 2 options
    kernel_initializers = ['glorot_normal' , 'glorot_uniform', 
                            'he_normal', 'he_uniform',
                            'lecun_normal', 'lecun_uniform']
    # 
    param_grid = dict(# kernel_initializer=kernel_initializers,
                      n_hidden_layers = n_hidden_layers,
                      intermediate_dim = intermediate_dims,
                      latent_dim = latent_dims,
                      latent_activation = latent_activations)
    # 
    grid = GridSearchCV(estimator = model, 
                        param_grid = param_grid, 
                        error_score = np.nan,
                        cv = n_cv)

if UseRandomizedSearchCV:
    search_type = 'randomized_'

    n_cv = 5
    n_RSCV_iters = 500 # Lower this if it takes more than 1 day
    n_hidden_layers = np.arange(1,6) 
    intermediate_dims = np.arange(1,8)*128
    latent_dims = np.arange(2,10) 
    latent_activations = ['relu', 'elu'] 
    kernel_initializers = ['glorot_normal' , 'glorot_uniform', 
                            'he_normal', 'he_uniform',
                            'lecun_normal', 'lecun_uniform']
    
    param_grid = dict(n_hidden_layers = n_hidden_layers,
                      # kernel_initializer = kernel_initializers,
                      intermediate_dim = intermediate_dims,
                      latent_dim = latent_dims,
                      latent_activation = latent_activations)

    grid = RandomizedSearchCV(estimator = model, 
                              param_distributions = param_grid, 
                              error_score = np.nan,
                              n_iter = n_RSCV_iters,
                              cv = n_cv)

grid_result = grid.fit(y_train, y_train)

time_stamp = int(time())
data_dir = '/SaveFiles/'
save_dir = os.environ['HOME'] + data_dir

savename_tmplt = save_dir + 'grid_vae_{}_{}'

if verbose: print("[INFO] Saving fitted model every way that I know how.")

# if verbose: print("[INFO] Saving full model")
# grid.save(savename_tmplt.format(time_stamp, 'full_model_save.hdf5'))

# if verbose: print("[INFO] Saving model weights")
# grid.save_weights(savename_tmplt.format(time_stamp, 
#                         'model_weights_save.hdf5'))

save_name_now = savename_tmplt.format(search_type, time_stamp, 
                                        'full_model.joblib.save')
if verbose: print("[INFO] Saving model instance to {}".format(save_name_now))
# joblib.dump(grid, save_name_now)


save_name_now = savename_tmplt.format(search_type, time_stamp, 
                                        'model_history.joblib.save')
if verbose: print("[INFO] Saving model history output to "
                    "{}".format(save_name_now))
# joblib.dump(grid_result, save_name_now)

# Summarize results
print("\n** Best Model Over Grid **")
print("{:18}: {}".format('Score', grid_result.best_score_))
for key, val in grid_result.best_params_.items():
    print("{:18}: {}".format(key,val))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("\nMean Test Score: {} +\\- {} ".format(mean, stdev))
    print('with: ')
    for key,val in param.items():
        print('\t{:18}: {}'.format(key,val))