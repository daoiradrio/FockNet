import torch

from NeuralNetworks.fockset import FockSet
from NeuralNetworks.focknet import FockNet



def train_model(model, train_dataset, train_params):
    pass



# Paths for training and validation data
train_annotation_file = "/Users/dario/datasets/H_sets/small_train_set/molslist.dat"
train_dftb_dir = "/Users/dario/datasets/H_sets/small_train_set/DFTB"
train_delta_dir = "/Users/dario/datasets/H_sets/small_train_set/DELTA"
train_xyz_dir = "/Users/dario/datasets/H_sets/small_train_set/XYZ"

valid_annotation_file = "/Users/dario/datasets/H_sets/small_valid_set/molslist.dat"
valid_dftb_dir = "/Users/dario/datasets/H_sets/small_valid_set/DFTB"
valid_delta_dir = "/Users/dario/datasets/H_sets/small_valid_set/DELTA"
valid_xyz_dir = "/Users/dario/datasets/H_sets/small_valid_set/XYZ"

# Initialize dataloaders for training and validation
train_dataset = FockSet(
    mols_list_file=train_annotation_file,
    dftb_dir=train_dftb_dir,
    delta_dir=train_delta_dir,
    xyz_dir=train_xyz_dir
)

valid_dataset = FockSet(
    mols_list_file=valid_annotation_file,
    dftb_dir=valid_dftb_dir,
    delta_dir=valid_delta_dir,
    xyz_dir=valid_xyz_dir
)

# Hyperparameters
n_features = 8

# Initialize model
focknet = FockNet(n_features=n_features)

# Training parameters
batch_size = 3
n_epochs = 10
learning_rate = 0.001
train_params = (batch_size, n_epochs, learning_rate)
