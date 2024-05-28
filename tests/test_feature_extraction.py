import numpy as np

from FockNet.fockset import FockSet
from FockNet.utils import benchmark_atom_features, atom_features, benchmark_pair_features, pair_features



num_test_samples = 2
tol = 1e-5
annotation_file = "/Users/dario/datasets/H_sets/small_train_set/molslist.dat"
dftb_dir = "/Users/dario/datasets/H_sets/small_train_set/DFTB"
rose_dir = "/Users/dario/datasets/H_sets/small_train_set/ROSE"
delta_dir = "/Users/dario/datasets/H_sets/small_train_set/DELTA"
xyz_dir = "/Users/dario/preprocessed_QM9/x_small_train_set"
fockset = FockSet(annotation_file, dftb_dir, rose_dir, delta_dir, xyz_dir)
num_mols = fockset.__len__()



def test_atom_feature_extraction():
    check_equivalence = []

    samples = np.random.choice([i for i in range(num_mols)], size=num_test_samples)

    for imol in samples:

        H_dftb, elems = fockset.__getitem__(imol)

        features_ref = benchmark_atom_features(H_dftb, elems)
        features = atom_features(H_dftb, elems)
        for f_ref, f in zip(features_ref, features):
            check_equivalence.append(np.allclose(f_ref, f, rtol=tol, atol=tol))
    
    assert all(check_equivalence)



def test_pair_feature_extraction():
    check_equivalence = []

    samples = np.random.choice([i for i in range(num_mols)], size=num_test_samples)

    for imol in samples:

        H_dftb, elems = fockset.__getitem__(imol)
        features_ref = benchmark_pair_features(H_dftb, elems)
        features = pair_features(H_dftb, elems)

        for f_ref, f in zip(features_ref, features):
            check_equivalence.append(np.allclose(f_ref, f, rtol=tol, atol=tol))
    
    assert all(check_equivalence)
