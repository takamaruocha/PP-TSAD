import numpy as np

# Dataset and path settings
data = 'TODS'  # Name of the dataset to be used
root_path = './datasets/'  # Root path for the dataset
saved_models_dir = './saved_models/' + data  # Directory for saving models
logs = 'logs/' + data + '/'  # Directory for saving logs

# Sequence settings
in_len = 50  # Length of the sequence
step_size = in_len  # Step size for creating sequences
win_size = in_len  # Window size for sequences
n_classes = 5  # Number of classes
anormly_ratio = 1.0  # Ratio for anomaly detection

# Model hyperparameters
d_model = 512  # Embedding dimension
d_ff = 512  # Feed-forward layer dimension
n_heads = 8  # Number of attention heads
e_layers = 3  # Number of encoder layers
dropout = 0.1  # Dropout rate

# fa model hyperparameters
seg_len_fa = 5  # Segment length for the FA model
factor_fa = 10  # Factor for the FA model
d_model_fa = 256  # Embedding dimension for the FA model
d_ff_fa = 512  # Feed-forward layer dimension for the FA model
n_heads_fa = 1  # Number of attention heads for the FA model
e_layers_fa = 3  # Number of encoder layers for the FA model
dropout_fa = 0.0  # Dropout rate for the FA model
attn_ratio_fa = 1.0  # Attention ratio for the FA model
patience = 5  # Patience for early stopping

# GPU settings
use_gpu = True  # Whether to use GPU
gpu = 2  # GPU ID to use
use_multi_gpu = True  # Whether to use multiple GPUs
devices = '2,3,4,5'  # List of GPU device IDs to use

# Dataset information
data_parser = {
    'ECG': {'data_dim': 1, 'split': [0.8, 0.2]},
    'TODS': {'data_dim': 5, 'split': [0.8, 0.2]},
    'SWaT': {'data_dim': 51, 'split': [0.8, 0.2]},
    'PSM': {'data_dim': 25, 'split': [0.8, 0.2]},
}

# Retrieve dataset settings
data_info = data_parser[data]  # Get information for the selected dataset
data_dim = data_info['data_dim']  # Dimension of the data
data_split = data_info['split']  # Split ratio for the data

# Data augmentation settings
jitter_ratio = 1.1  # Ratio for jittering
jitter_scale_ratio = 0.8  # Scaling ratio for jittering
max_seg = 8  # Maximum number of segments

# Context_Cont settings
temperature = 0.2  # Temperature parameter for contrastive learning
use_cosine_similarity = True  # Whether to use cosine similarity

# Training settings
batch_size = 32 * 4  # Batch size
learning_rate = 1e-4  # Learning rate
num_epochs = 10  # Number of epochs

# Loss weights and learning rates for various components
ft_loss_weight = 1  # Weight for the feature extraction loss

learning_rate_fa = 0.1 * learning_rate  # Learning rate for the FA model
learning_rate_fb = 0.1 * learning_rate  # Learning rate for the FB model (if applicable)
learning_rate_ft = learning_rate  # Learning rate for the FT model
learning_rate_fc = learning_rate  # Learning rate for the FC model

