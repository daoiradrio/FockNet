import os

import numpy as np

from CompChemUtils.files import read_xyz_file
from torch.utils.data import Dataset



class FockSet(Dataset):

    def __init__(self, mols_list_file, dftb_dir, delta_dir, xyz_dir):
        self.mols = np.loadtxt(mols_list_file, dtype=str)
        self.dftb_dir = dftb_dir
        self.delta_dir = delta_dir
        self.xyz_dir = xyz_dir


    def __len__(self) -> int:
        return self.mols.size
    

    def __getitem__(self, idx) -> (np.array, np.array):
        dftb_path = os.path.join(self.dftb_dir, f"DFTB_Fock_{self.mols[idx]}.dat")
        dftb_fock = np.loadtxt(dftb_path)
        delta_path = os.path.join(self.delta_dir, f"DELTA_Fock_{self.mols[idx]}.dat")
        delta_fock = np.loadtxt(delta_path)
        xyz_path = os.path.join(self.xyz_dir, f"{self.mols[idx]}.xyz")
        xyz_data = read_xyz_file(xyz_path)
        naos = int(np.sqrt(dftb_fock.size))
        dftb_fock.reshape((naos, naos), order="F")
        delta_fock.reshape((naos, naos))
        return dftb_fock, delta_fock, xyz_data
