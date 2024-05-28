import jax
import optax

from fockset import FockSet
from focknet import FockNet
from torch.utils.data import DataLoader



def custom_collate_fn(batch):
    batch_H_dftb, batch_H_delta, batch_xyz = zip(*batch)



def train_model(model, train_dataset, valid_dataset, train_params):
    batch_size, n_epochs, learning_rate = train_params

    init_atom_features, init_pair_features, init_pair_split, _ = valid_dataset.__getitem__(0)
    params = model.init(
        jax.random.PRNGKey(0),
        init_atom_features,
        init_pair_features,
        init_pair_split
    )
    
    '''
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    '''

    n = train_dataset.__len__()
    for i in range(n):
        atom_features, pair_features, pair_split, _ = train_dataset.__getitem__(i)
        out = model.apply(params, atom_features, pair_features, pair_split)
        break






# Paths for training and validation data
train_annotation_file = "/Users/dario/datasets/H_sets/small_train_set/molslist.dat"
train_dftb_dir = "/Users/dario/datasets/H_sets/small_train_set/DFTB"
train_rose_dir = "/Users/dario/datasets/H_sets/small_train_set/ROSE"
train_delta_dir = "/Users/dario/datasets/H_sets/small_train_set/DELTA"
train_xyz_dir = "/Users/dario/preprocessed_QM9/x_small_train_set"

valid_annotation_file = "/Users/dario/datasets/H_sets/small_valid_set/molslist.dat"
valid_dftb_dir = "/Users/dario/datasets/H_sets/small_valid_set/DFTB"
valid_rose_dir = "/Users/dario/datasets/H_sets/small_valid_set/ROSE"
valid_delta_dir = "/Users/dario/datasets/H_sets/small_valid_set/DELTA"
valid_xyz_dir = "/Users/dario/preprocessed_QM9/x_small_valid_set"

# Initialize dataloaders for training and validation
train_dataset = FockSet(
    mols_list_file=train_annotation_file,
    dftb_dir=train_dftb_dir,
    rose_dir=train_rose_dir,
    delta_dir=train_delta_dir,
    xyz_dir=train_xyz_dir
)

valid_dataset = FockSet(
    mols_list_file=valid_annotation_file,
    dftb_dir=valid_dftb_dir,
    rose_dir=valid_rose_dir,
    delta_dir=valid_delta_dir,
    xyz_dir=valid_xyz_dir
)

# Hyperparameters
num_features = 8
num_blocks = 2

# Initialize model
focknet = FockNet(num_features=num_features, num_blocks=num_blocks)

# Training parameters
batch_size = 3
n_epochs = 10
learning_rate = 0.001
train_params = (batch_size, n_epochs, learning_rate)

train_model(focknet, train_dataset, valid_dataset, train_params)
