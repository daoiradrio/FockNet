import os
import e3x

import numpy as np

from CompChemUtils.files import read_xyz_file
from torch.utils.data import Dataset



class FockSet(Dataset):

    def __init__(self, mols_list_file, dftb_dir, rose_dir, delta_dir, xyz_dir):
        self.mols = np.loadtxt(mols_list_file, usecols=1, dtype=str)
        self.dftb_dir = dftb_dir
        self.rose_dir = rose_dir
        self.delta_dir = delta_dir
        self.xyz_dir = xyz_dir


    def __len__(self):
        return self.mols.size
    

    def __getitem__(self, idx):
        dftb_path = os.path.join(self.dftb_dir, f"DFTB_Fock_{self.mols[idx]}.dat")
        H_dftb = np.loadtxt(dftb_path)
        #rose_path = os.path.join(self.rose_dir, f"ROSE_Fock_{self.mols[idx]}.dat")
        #H_rose = np.loadtxt(rose_path)
        delta_path = os.path.join(self.delta_dir, f"DELTA_Fock_{self.mols[idx]}.dat")
        H_delta = np.loadtxt(delta_path)
        xyz_path = os.path.join(self.xyz_dir, f"{self.mols[idx]}.xyz")
        _, elems, _ = read_xyz_file(xyz_path)

        naos = int(np.sqrt(H_dftb.size))
        
        H_dftb = H_dftb.reshape((naos, naos))
        #H_rose = H_rose.reshape((naos, naos))
        H_delta = H_delta.reshape((naos, naos))

        #atom_features, pair_features = get_atom_and_pair_features(H_dftb, elems)

        pair_split, _ = e3x.ops.sparse_pairwise_indices(len(elems))
        pair_split = np.array(pair_split)

        #return atom_features, pair_features, pair_split, H_delta
        return H_dftb, elems
