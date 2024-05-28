from fockset import FockSet
import numpy as np
from e3x.so3.irreps import clebsch_gordan, clebsch_gordan_for_degrees
from time import time
from CompChemUtils.visual import plotmat
from CompChemUtils.chemdata import NAO_minimal_valence
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



def scaling(x, m, n):
    return m * x**n



def matrix_element_correlation():

    from CompChemUtils.structure import Structure
    from utils import get_ao_indices
    from CompChemUtils.visual import printmat

    annotation_file = "/Users/dario/datasets/H_sets/small_train_set/molslist.dat"
    #dftb_dir = "/Users/dario/datasets/H_sets/small_train_set/DFTB"
    #rose_dir = "/Users/dario/datasets/H_sets/small_train_set/ROSE"
    dftb_dir = "/Users/dario/datasets/H_sets/correlation_test_set/DFTB"
    rose_dir = "/Users/dario/datasets/H_sets/correlation_test_set/ROSE"
    delta_dir = "/Users/dario/datasets/H_sets/small_train_set/DELTA"
    xyz_dir = "/Users/dario/preprocessed_QM9/x_small_train_set"

    fockset = FockSet(annotation_file, dftb_dir, rose_dir, delta_dir, xyz_dir)
    num_mols = fockset.__len__()

    molstruc = Structure()
    
    H_dftb_submats = []
    H_rose_submats = []
    for mol in range(num_mols):
        H_dftb, H_rose, elems, coords = fockset.__getitem__(mol)
        molstruc.set_structure(elems, coords, as_matrix=False)
        ao_indices = get_ao_indices(elems)
        H_dftb_submat = np.zeros((5,5))
        H_rose_submat = np.zeros((5,5))
        for idx1 in range(molstruc.natoms):
            if molstruc.elems[idx1] == "O":
                if len(molstruc.bond_dict[idx1]) == 2:
                    for idx2 in molstruc.bond_dict[idx1]:
                        if molstruc.elems[idx2] == "H":
                            O_label = f"{molstruc.elems[idx1]}{idx1}"
                            H_label = f"{molstruc.elems[idx2]}{idx2}"
                            H_idx = ao_indices[H_label]["s"]
                            O_idx = ao_indices[O_label]["s"][0], ao_indices[O_label]["p"][1]
                            H_dftb_submat[0:1, 0:1] = H_dftb[H_idx[0]:H_idx[1], H_idx[0]:H_idx[1]]
                            H_dftb_submat[1:5, 1:5] = H_dftb[O_idx[0]:O_idx[1], O_idx[0]:O_idx[1]]
                            H_dftb_submat[0:1, 1:5] = H_dftb[H_idx[0]:H_idx[1], O_idx[0]:O_idx[1]]
                            H_dftb_submat[1:5, 0:1] = H_dftb[O_idx[0]:O_idx[1], H_idx[0]:H_idx[1]]
                            H_dftb_submats.append(H_dftb_submat)
                            H_rose_submat[0:1, 0:1] = H_rose[H_idx[0]:H_idx[1], H_idx[0]:H_idx[1]]
                            H_rose_submat[1:5, 1:5] = H_rose[O_idx[0]:O_idx[1], O_idx[0]:O_idx[1]]
                            H_rose_submat[0:1, 1:5] = H_rose[H_idx[0]:H_idx[1], O_idx[0]:O_idx[1]]
                            H_rose_submat[1:5, 0:1] = H_rose[O_idx[0]:O_idx[1], H_idx[0]:H_idx[1]]
                            H_rose_submats.append(H_rose_submat)
    H_dftb_submats = np.stack(H_dftb_submats)
    H_rose_submats = np.stack(H_rose_submats)

    corrmat = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            X_dftb = H_dftb_submats[:, i, j]
            X_rose = H_rose_submats[:, i, j]
            corr = np.corrcoef(X_dftb, X_rose)
            corrmat[i][j] = corr[0][1]
            #print(f"{i}, {j}:")
            #printmat(corr)
            #X_dftb_mean = np.mean(X_dftb)
            #X_rose_mean = np.mean(X_rose)
            #X_dftb_std = np.std(X_dftb)
            #X_rose_std = np.std(X_rose)
            #covmat = np.cov(X_dftb, X_rose)
            #covmat_norm = covmat / (X_dftb_std * X_rose_std)
    #plotmat(corrmat)
    
    H_dftb_submats = []
    H_rose_submats = []
    for mol in range(num_mols):
        H_dftb, H_rose, elems, coords = fockset.__getitem__(mol)
        molstruc.set_structure(elems, coords, as_matrix=False)
        ao_indices = get_ao_indices(elems)
        H_dftb_submat = np.zeros((6,6))
        H_rose_submat = np.zeros((6,6))
        for idx1 in range(molstruc.natoms):
            if molstruc.elems[idx1] == "N":
                    H_count = 0
                    H_idx = []
                    for idx2 in molstruc.bond_dict[idx1]:
                        if molstruc.elems[idx2] == "H":
                            H_count += 1
                            H_idx.append(idx2)
                    if H_count == 2:
                        N_label = f"N{idx1}"
                        H1_label = f"H{H_idx[0]}"
                        H2_label = f"H{H_idx[1]}"
                        N_idx = ao_indices[N_label]["s"][0], ao_indices[N_label]["p"][1]
                        H1_idx = ao_indices[H1_label]["s"]
                        H2_idx = ao_indices[H2_label]["s"]
                        H_dftb_submat[0:1, 0:1] = H_dftb[H1_idx[0]:H1_idx[1], H1_idx[0]:H1_idx[1]]
                        H_dftb_submat[1:2, 1:2] = H_dftb[H2_idx[0]:H2_idx[1], H2_idx[0]:H2_idx[1]]
                        H_dftb_submat[2:6, 2:6] = H_dftb[N_idx[0]:N_idx[1], N_idx[0]:N_idx[1]]
                        H_dftb_submat[0:1, 1:2] = H_dftb[H1_idx[0]:H1_idx[1], H2_idx[0]:H2_idx[1]]
                        H_dftb_submat[1:2, 0:1] = H_dftb[H2_idx[0]:H2_idx[1], H1_idx[0]:H1_idx[1]]
                        H_dftb_submat[0:1, 2:6] = H_dftb[H1_idx[0]:H1_idx[1], N_idx[0]:N_idx[1]]
                        H_dftb_submat[2:6, 0:1] = H_dftb[N_idx[0]:N_idx[1], H1_idx[0]:H1_idx[1]]
                        H_dftb_submat[1:2, 2:6] = H_dftb[H2_idx[0]:H2_idx[1], N_idx[0]:N_idx[1]]
                        H_dftb_submat[2:6, 1:2] = H_dftb[N_idx[0]:N_idx[1], H2_idx[0]:H2_idx[1]]
                        H_dftb_submats.append(H_dftb_submat)
                        H_rose_submat[0:1, 0:1] = H_rose[H1_idx[0]:H1_idx[1], H1_idx[0]:H1_idx[1]]
                        H_rose_submat[1:2, 1:2] = H_rose[H2_idx[0]:H2_idx[1], H2_idx[0]:H2_idx[1]]
                        H_rose_submat[2:6, 2:6] = H_rose[N_idx[0]:N_idx[1], N_idx[0]:N_idx[1]]
                        H_rose_submat[0:1, 1:2] = H_rose[H1_idx[0]:H1_idx[1], H2_idx[0]:H2_idx[1]]
                        H_rose_submat[1:2, 0:1] = H_rose[H2_idx[0]:H2_idx[1], H1_idx[0]:H1_idx[1]]
                        H_rose_submat[0:1, 2:6] = H_rose[H1_idx[0]:H1_idx[1], N_idx[0]:N_idx[1]]
                        H_rose_submat[2:6, 0:1] = H_rose[N_idx[0]:N_idx[1], H1_idx[0]:H1_idx[1]]
                        H_rose_submat[1:2, 2:6] = H_rose[H2_idx[0]:H2_idx[1], N_idx[0]:N_idx[1]]
                        H_rose_submat[2:6, 1:2] = H_rose[N_idx[0]:N_idx[1], H2_idx[0]:H2_idx[1]]
                        H_rose_submats.append(H_rose_submat)
    H_dftb_submats = np.stack(H_dftb_submats)
    H_rose_submats = np.stack(H_rose_submats)

    corrmat = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            X_dftb = H_dftb_submats[:, i, j]
            X_rose = H_rose_submats[:, i, j]
            corr = np.corrcoef(X_dftb, X_rose)
            corrmat[i][j] = corr[0][1]
            #print(f"{i}, {j}:")
            #printmat(corr)
            #X_dftb_mean = np.mean(X_dftb)
            #X_rose_mean = np.mean(X_rose)
            #X_dftb_std = np.std(X_dftb)
            #X_rose_std = np.std(X_rose)
            #covmat = np.cov(X_dftb, X_rose)
            #covmat_norm = covmat / (X_dftb_std * X_rose_std)
    plotmat(corrmat)



def full_feature_extraction():
    annotation_file = "/Users/dario/datasets/H_sets/small_train_set/molslist.dat"
    dftb_dir = "/Users/dario/datasets/H_sets/small_train_set/DFTB"
    rose_dir = "/Users/dario/datasets/H_sets/small_train_set/ROSE"
    delta_dir = "/Users/dario/datasets/H_sets/small_train_set/DELTA"
    xyz_dir = "/Users/dario/preprocessed_QM9/x_small_train_set"

    fockset = FockSet(annotation_file, dftb_dir, rose_dir, delta_dir, xyz_dir)
    n = fockset.__len__()

    CG = clebsch_gordan(1, 1, 2)
    CG000 = clebsch_gordan_for_degrees(0, 0, 0)
    CG011 = clebsch_gordan_for_degrees(0, 1, 1)
    CG101 = clebsch_gordan_for_degrees(1, 0, 1)
    CG110 = clebsch_gordan_for_degrees(1, 1, 0)
    CG111 = clebsch_gordan_for_degrees(1, 1, 1)
    CG112 = clebsch_gordan_for_degrees(1, 1, 2)

    timing_benchmark_atoms = 0
    timing_benchmark_pairs = 0
    timing_einsum_atoms = 0
    timing_einsum_pairs = 0

    Ns = []
    ts = []
    
    for j in range(n):
        #if j == 50:
        #    break
        print(j)

        #H_dftb, _, elems, _ = fockset.__getitem__(j)
        H_dftb, elems= fockset.__getitem__(j)
        eps, C = np.linalg.eig(H_dftb)

        t_start = time()

        sort_idx = np.argsort(eps)[::-1]
        C = C[:, sort_idx]
        eps = eps[sort_idx]
        C_eps = np.dot(C, np.diag(eps))
        num_mos = C.shape[1]
        num_atoms = len(elems)
        iao1 = 0
        ipair = 0
        
        atom_features_benchmark = np.zeros((num_atoms, 1, 9, 1))
        atom_features_einsum = np.zeros((num_atoms, 1, 9, 1))

        pair_features_benchmark = np.zeros((int(num_atoms*(num_atoms-1)/2), 1, 9, 1))
        pair_features_einsum = np.zeros((int(num_atoms*(num_atoms-1)/2), 1, 9, 1))

        # NEEDED FOR OLD ONE SHOT EINSUM
        '''
        c_vecs = np.zeros((num_atoms, 4, num_mos))
        c_eps_vecs = np.zeros((num_atoms, 4, num_mos))
        iao = 0
        for ielem, elem in enumerate(elems):
            c_vecs[ielem, 0, :] = C[iao, :]
            c_eps_vecs[ielem, 0, :] = C_eps[iao, :]
            iao += 1
            if elem != "H":
                c_vecs[ielem, 1:, :] = C[iao:iao+3, :]
                c_eps_vecs[ielem, 1:, :] = C_eps[iao:iao+3, :]
                iao += 3
        '''

        Ns.append(num_atoms)

        for ielem1, elem1 in enumerate(elems):
            iao2 = 0

            #start_einsum_atoms = time()

            # ONE SHOT EINSUM
            if elem1 == "H":
                atom_features_einsum[ielem1, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao1, :])
            else:
                atom_features_einsum[ielem1, 0, :, 0] = np.einsum("abc, an, bn -> c", CG, C[iao1 : iao1+4, :], C_eps[iao1 : iao1+4, :])
            
            #timing_einsum_atoms += time() - start_einsum_atoms

            #'''
            start_benchmark_atoms = time()

            # MANUAL BENCHMARK
            # l3 = 0
            for i in range(num_mos):
                atom_features_benchmark[ielem1, 0, 0, 0] += (CG000[0, 0, 0] * C[iao1, i] * C_eps[iao1, i])
            if elem1 != "H":
                for m1 in range(3):
                    for m2 in range(3):
                        for i in range(num_mos):
                            atom_features_benchmark[ielem1, 0, 0, 0] += (CG110[m1, m2, 0] * C[iao1+1+m1, i] * C_eps[iao1+1+m2, i])
            # l3 = 1
                for m3 in range(3):
                    for m1 in range(3):
                        for i in range(num_mos):
                            atom_features_benchmark[ielem1, 0, 1+m3, 0] += (CG101[m1, 0, m3] * C[iao1+1+m1, i] * C_eps[iao1, i])
                for m3 in range(3):
                    for m2 in range(3):
                        for i in range(num_mos):
                            atom_features_benchmark[ielem1, 0, 1+m3, 0] += (CG011[0, m2, m3] * C[iao1, i] * C_eps[iao1+1+m2, i])
                for m3 in range(3):
                    for m1 in range(3):
                        for m2 in range(3):
                            for i in range(num_mos):
                                atom_features_benchmark[ielem1, 0, 1+m3, 0] += (CG111[m1, m2, m3] * C[iao1+1+m1, i] * C_eps[iao1+1+m2, i])
            # l3 = 2
                for m3 in range(5):
                    for m1 in range(3):
                        for m2 in range(3):
                            for i in range(num_mos):
                                atom_features_benchmark[ielem1, 0, 4+m3, 0] += (CG112[m1, m2, m3] * C[iao1+1+m1, i] * C_eps[iao1+1+m2, i])
            
            timing_benchmark_atoms += time() - start_benchmark_atoms

            #print()
            #print(timing_benchmark)
            #print(timing_one_shot)
            #print(timing_fixed_one_shot)
            #print()

            #print(np.allclose(atom_features_benchmark[ielem1, :, :, :], atom_features_fixed_one_shot[ielem1, :, :, :], rtol=1e-6, atol=1e-6))
            #print(np.allclose(atom_features_one_shot[ielem1, :, :, :], atom_features_fixed_one_shot[ielem1, :, :, :], rtol=1e-6, atol=1e-6))

            '''
            # l3=0 test
            if not np.allclose(atom_features_benchmark[ielem1, 0, 0, 0], atom_features_one_shot[ielem1, 0, 0, 0], rtol=1e-6, atol=1e-6):
                print()
                print("l3 = 0 Test")
                print(atom_features_benchmark[ielem1, 0, 0, 0])
                print(atom_features_one_shot[ielem1, 0, 0, 0])
                print()
            # l3=1 test
            if not np.allclose(atom_features_benchmark[ielem1, 0, 1:4, 0], atom_features_one_shot[ielem1, 0, 1:4, 0], rtol=1e-6, atol=1e-6):
                print()
                print("l3 = 1 Test")
                print(atom_features_benchmark[ielem1, 0, 1:4, 0])
                print(atom_features_one_shot[ielem1, 0, 1:4, 0])
                print()
            # l3=2 test
            if not np.allclose(atom_features_benchmark[ielem1, 0, 4:, 0], atom_features_one_shot[ielem1, 0, 4:, 0], rtol=1e-6, atol=1e-6):
                print()
                print("l3 = 2 Test")
                print(atom_features_benchmark[ielem1, 0, 4:, 0])
                print(atom_features_one_shot[ielem1, 0, 4:, 0])
                print()
            '''

            for ielem2, elem2 in enumerate(elems[:ielem1]):
                #start_einsum_pairs = time()

                # ONE SHOT EINSUM
                if elem1 == "H" and elem2 == "H":
                    pair_features_einsum[ipair, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao2, :])
                elif elem1 != "H" and elem2 != "H":
                    pair_features_einsum[ipair, 0, :, 0] = np.einsum("abc, an, bn -> c", CG, C[iao1 : iao1+4, :], C_eps[iao2 : iao2+4, :])
                elif elem1 == "H" and elem2 != "H":
                    pair_features_einsum[ipair, 0, :4, 0] = np.dot(C[iao1, :], C_eps[iao2 : iao2+4, :].T)
                elif elem1 != "H" and elem2 == "H":
                    pair_features_einsum[ipair, 0, :4, 0] = np.dot(C[iao1 : iao1+4, :], C_eps[iao2, :])
                
                #timing_einsum_pairs += time() - start_einsum_pairs

                #'''
                start_benchmark_pairs = time()

                # MANUAL BENCHMARK
                # l3 = 0
                for i in range(num_mos):
                    pair_features_benchmark[ipair, 0, 0, 0] += (C[iao1, i] * C_eps[iao2, i])
                if elem1 != "H" and elem2 != "H":
                    for m1 in range(3):
                        for m2 in range(3):
                            for i in range(num_mos):
                                pair_features_benchmark[ipair, 0, 0, 0] += (CG110[m1, m2, 0] * C[iao1+1+m1, i] * C_eps[iao2+1+m2, i])
                # l3 = 1
                if elem1 != "H" and elem2 == "H":
                    for m3 in range(3):
                        for m1 in range(3):
                            for i in range(num_mos):
                                pair_features_benchmark[ipair, 0, 1+m3, 0] += (CG101[m1, 0, m3] * C[iao1+1+m1, i] * C_eps[iao2, i])
                if elem1 == "H" and elem2 != "H":
                    for m3 in range(3):
                        for m2 in range(3):
                            for i in range(num_mos):
                                pair_features_benchmark[ipair, 0, 1+m3, 0] += (CG011[0, m2, m3] * C[iao1, i] * C_eps[iao2+1+m2, i])
                if elem1 != "H" and elem2 != "H":
                    for m3 in range(3):
                        for m1 in range(3):
                            for m2 in range(3):
                                for i in range(num_mos):
                                    pair_features_benchmark[ipair, 0, 1+m3, 0] += (CG111[m1, m2, m3] * C[iao1+1+m1, i] * C_eps[iao2+1+m2, i])
                            for i in range(num_mos):
                                pair_features_benchmark[ipair, 0, 1+m3, 0] += (CG101[m1, 0, m3] * C[iao1+1+m1, i] * C_eps[iao2, i])
                        for m2 in range(3):
                            for i in range(num_mos):
                                pair_features_benchmark[ipair, 0, 1+m3, 0] += (CG011[0, m2, m3] * C[iao1, i] * C_eps[iao2+1+m2, i])
                # l3 = 2
                    for m3 in range(5):
                        for m1 in range(3):
                            for m2 in range(3):
                                for i in range(num_mos):
                                    pair_features_benchmark[ipair, 0, 4+m3, 0] += (CG112[m1, m2, m3] * C[iao1+1+m1, i] * C_eps[iao2+1+m2, i])
                
                timing_benchmark_pairs += time() - start_benchmark_pairs
                
                #if not np.allclose(pair_features_benchmark[ipair, 0, :, 0], pair_features_einsum[ipair, 0, :, 0], rtol=1e-6, atol=1e-6):
                #    print("falsch")
                #    print()
                #'''

                iao2 += NAO_minimal_valence[elem2]
                ipair += 1
            
            iao1 += NAO_minimal_valence[elem1]

        print(np.allclose(atom_features_benchmark, atom_features_einsum, rtol=1e-6, atol=1e-6))
        print(np.allclose(pair_features_benchmark, pair_features_einsum, rtol=1e-6, atol=1e-6))
        
        ts.append(time() - t_start)

    Ns, ts = zip(*sorted(zip(Ns, ts)))
    collect_ts = [ts[0]]
    plot_Ns = [Ns[0]]
    plot_ts = []
    for N, t in zip(Ns[1:], ts[1:]):
        if N == plot_Ns[-1]:
            collect_ts.append(t)
        else:
            plot_ts.append(np.mean(collect_ts))
            collect_ts = [t]
            plot_Ns.append(N)
    plot_ts.append(np.mean(collect_ts))
    params_opt, _ = curve_fit(scaling, plot_Ns, plot_ts)
    #print(f"{params_opt[0]:.5f}\t{params_opt[1]:.2f}")
    plt.plot(plot_Ns, plot_ts, label="Measured Timings")
    plt.plot(plot_Ns, scaling(plot_Ns, params_opt[0], params_opt[1]), label=f"Fitted Scaling ({params_opt[0]:.5f} * x^{params_opt[1]:.2f})")
    plt.xticks(np.linspace(plot_Ns[0], plot_Ns[-1], plot_Ns[-1]-plot_Ns[0]+1))
    plt.xlim(plot_Ns[0], plot_Ns[-1])
    plt.xlabel("Number of Atoms")
    #plt.ylim(min(plot_ts), max(plot_ts))
    plt.ylabel("Wall Time [s]")
    plt.grid()
    plt.legend()
    #plt.show()
    
    '''
    print()
    print(timing_benchmark_atoms / (j+1))
    print(timing_benchmark_pairs / (j+1))
    print()
    print(timing_einsum_atoms / (j+1))
    print(timing_einsum_pairs / (j+1))
    print()
    '''



def approx_feature_extraction():
    annotation_file = "/Users/dario/datasets/H_sets/small_train_set/molslist.dat"
    dftb_dir = "/Users/dario/datasets/H_sets/small_train_set/DFTB"
    rose_dir = "/Users/dario/datasets/H_sets/small_train_set/ROSE"
    delta_dir = "/Users/dario/datasets/H_sets/small_train_set/DELTA"
    xyz_dir = "/Users/dario/preprocessed_QM9/x_small_train_set"

    r_cut = 10.0

    fockset = FockSet(annotation_file, dftb_dir, rose_dir, delta_dir, xyz_dir)
    n = fockset.__len__()

    CG = clebsch_gordan(1, 1, 2)
    CG011 = clebsch_gordan_for_degrees(0, 1, 1)
    CG101 = clebsch_gordan_for_degrees(1, 0, 1)

    Ns = []
    ts = []

    for i in range(n):
        #if i == 50:
        #    break
        print(i)

        H_dftb, _, elems, _ = fockset.__getitem__(i)

        t_start = time()

        num_atoms = len(elems)
        #atom_features = np.zeros((num_atoms, 1, 9, 1))
        pair_features = np.zeros((int(num_atoms*(num_atoms-1)/2), 1, 9, 1))
        ipair = 0
        iao1 = 0

        Ns.append(num_atoms)

        for ielem1, elem1 in enumerate(elems):

            nao1 = NAO_minimal_valence[elem1]
            iao2 = 0

            for elem2 in elems[:ielem1]:

                #d = np.linalg.norm(coords[ielem1] - coords[ielem2])
                #if d > r_cut:
                #    continue

                nao2 = NAO_minimal_valence[elem2]

                H_count = f"{elem1}{elem2}".count("H")
                if H_count == 2:
                    submat = np.zeros((2, 2))
                elif H_count == 1:
                    submat = np.zeros((5, 5))
                elif H_count == 0:
                    submat = np.zeros((8, 8))
                submat[0 : nao1, 0 : nao1] = H_dftb[iao1 : iao1+nao1, iao1 : iao1+nao1]
                submat[nao1 : nao1+nao2, nao1 : nao1+nao2] = H_dftb[iao2 : iao2+nao2, iao2 : iao2+nao2]
                submat[0 : nao1, nao1 : nao1+nao2] = H_dftb[iao1 : iao1+nao1, iao2 : iao2+nao2]
                submat[nao1 : nao1+nao2, 0 : nao1] = H_dftb[iao2 : iao2+nao2, iao1 : iao1+nao1]
                
                eps, C = np.linalg.eig(submat)
                sort_idx = np.argsort(eps)[::-1]
                C = C[:, sort_idx]
                C_eps = np.dot(C, np.diag(eps))

                if elem1 == "H" and elem2 == "H":
                    pair_features[ipair, 0, 0, 0] = np.dot(C[0, :], C_eps[1, :])
                elif elem1 == "H" and elem2 != "H":
                    pair_features[ipair, 0, 0, 0] = np.dot(C[0, :], C_eps[1, :])
                    #pair_features_einsum[ipair, 0, 1:4, 0] = CG011[0, 0, 0] * np.einsum("m, am -> a", C[iao1, :], C_eps[iao2+1 : iao2+4, :])
                    pair_features[ipair, 0, 1:4, 0] = CG011[0, 0, 0] * np.dot(C[0, :], C_eps[1:4, :].T)
                elif elem1 != "H" and elem2 == "H":
                    pair_features[ipair, 0, 0, 0] = np.dot(C[0, :], C_eps[4, :])
                    #pair_features_einsum[ipair, 0, 1:4, 0] = CG101[0, 0, 0] * np.einsum("am, m -> a", C[iao1+1 : iao1+4, :], C_eps[iao2, :])
                    pair_features[ipair, 0, 1:4, 0] = CG101[0, 0, 0] * np.dot(C[1:4, :], C_eps[4, :])
                elif elem1 != "H" and elem2 != "H":
                    pair_features[ipair, 0, :, 0] = np.einsum("abc, an, bn -> c", CG, C[:4, :], C_eps[4:, :])
 
                ipair += 1
                iao2 += nao2
            iao1 += nao1
        
        ts.append(time() - t_start)

    Ns, ts = zip(*sorted(zip(Ns, ts)))
    collect_ts = [ts[0]]
    plot_Ns = [Ns[0]]
    plot_ts = []
    for N, t in zip(Ns[1:], ts[1:]):
        if N == plot_Ns[-1]:
            collect_ts.append(t)
        else:
            plot_ts.append(np.mean(collect_ts))
            collect_ts = [t]
            plot_Ns.append(N)
    plot_ts.append(np.mean(collect_ts))
    params_opt, _ = curve_fit(scaling, plot_Ns, plot_ts)
    #print(f"{params_opt[0]:.5f}\t{params_opt[1]:.2f}")
    plt.plot(plot_Ns, plot_ts, label="Measured Timings")
    plt.plot(plot_Ns, scaling(plot_Ns, params_opt[0], params_opt[1]), label=f"Fitted Scaling ({params_opt[0]:.5f} * x^{params_opt[1]:.2f})")
    plt.xticks(np.linspace(plot_Ns[0], plot_Ns[-1], plot_Ns[-1]-plot_Ns[0]+1))
    plt.xlim(plot_Ns[0], plot_Ns[-1])
    plt.xlabel("Number of Atoms")
    #plt.ylim(min(plot_ts), max(plot_ts))
    plt.ylabel("Wall Time [s]")
    plt.grid()
    plt.legend()
    plt.show()



def decompose_and_compose_tensorproduct():
    from utils import get_atom_and_pair_features, get_fock_matrix

    annotation_file = "/Users/dario/datasets/H_sets/small_train_set/molslist.dat"
    dftb_dir = "/Users/dario/datasets/H_sets/small_train_set/DFTB"
    rose_dir = "/Users/dario/datasets/H_sets/small_train_set/ROSE"
    delta_dir = "/Users/dario/datasets/H_sets/small_train_set/DELTA"
    xyz_dir = "/Users/dario/preprocessed_QM9/x_small_train_set"

    fockset = FockSet(annotation_file, dftb_dir, rose_dir, delta_dir, xyz_dir)
    n = fockset.__len__()

    for i in range(n):
        print(i)

        H, elems = fockset.__getitem__(i)
        s_atom_features, p_atom_features, s_pair_features, p_pair_features = get_atom_and_pair_features(H, elems)
        #get_fock_matrix(s_atom_features, p_atom_features, s_pair_features, p_pair_features, elems, H.shape[0], H)

        break



if __name__ == "__main__":
    #full_feature_extraction()
    #approx_feature_extraction()
    decompose_and_compose_tensorproduct()